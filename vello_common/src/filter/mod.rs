// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Common filter helper functions.
//!
//! Unlike the filters defines in [`crate::filter_effects`], the filters in this module
//! represent a special representation of each filter to be used as the basis for rendering in
//! `vello_gpu` and `vello_cpu`.

use crate::filter::drop_shadow::{DropShadow, transform_shadow_params};
use crate::filter::flood::Flood;
use crate::filter::gaussian_blur::{GaussianBlur, transform_blur_params};
use crate::filter::offset::Offset;
use crate::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use crate::geometry::{PaddingU16, PaddingU32, RectU16, SizeU16};
use crate::kurbo::{Affine, Rect, Vec2};
use crate::math::snap_up;
use crate::tile::Tile;
use crate::util::RectExt;

pub mod drop_shadow;
pub mod flood;
pub mod gaussian_blur;
pub mod offset;

/// A filter that has been prepared for rendering.
#[derive(Debug)]
pub enum PreparedFilter {
    /// A flood filter.
    Flood(Flood),
    /// A gaussian blur filter.
    GaussianBlur(GaussianBlur),
    /// An offset filter.
    Offset(Offset),
    /// A drop shadow filter.
    DropShadow(DropShadow),
}

impl PreparedFilter {
    /// Build a new prepared filter for the given transform.
    pub fn new(filter: &Filter, transform: &Affine) -> Self {
        validate_filter_parameters(filter, transform);
        // Multi-primitive filter graphs are not yet implemented.
        if filter.graph.primitives.len() != 1 {
            unimplemented!("Multi-primitive filter graphs are not yet supported");
        }

        match &filter.graph.primitives[0] {
            FilterPrimitive::Flood { color } => {
                let flood = Flood::new(*color);
                Self::Flood(flood)
            }
            FilterPrimitive::GaussianBlur {
                std_deviation,
                edge_mode,
            } => {
                let scaled_std_dev = transform_blur_params(*std_deviation, transform);
                let blur = GaussianBlur::new(scaled_std_dev, *edge_mode);
                Self::GaussianBlur(blur)
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode,
            } => {
                let (scaled_dx, scaled_dy, scaled_std_dev) =
                    transform_shadow_params(*dx, *dy, *std_deviation, transform);
                let drop_shadow =
                    DropShadow::new(scaled_dx, scaled_dy, scaled_std_dev, *edge_mode, *color);

                Self::DropShadow(drop_shadow)
            }
            FilterPrimitive::DropShadowOnly {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode,
            } => {
                let (scaled_dx, scaled_dy, scaled_std_dev) =
                    transform_shadow_params(*dx, *dy, *std_deviation, transform);
                let drop_shadow = DropShadow::new_shadow_only(
                    scaled_dx,
                    scaled_dy,
                    scaled_std_dev,
                    *edge_mode,
                    *color,
                );

                Self::DropShadow(drop_shadow)
            }
            FilterPrimitive::Offset { dx, dy } => {
                let (scaled_dx, scaled_dy) = transform_offset_params(*dx, *dy, transform);
                let offset = Offset::new(scaled_dx, scaled_dy);

                Self::Offset(offset)
            }
            _ => {
                // Other primitives like Blend, ColorMatrix, ComponentTransfer, etc.
                // are not yet implemented
                unimplemented!("Other filter primitives not yet implemented");
            }
        }
    }
}

/// Metadata about a filter layer and how it should be composited back into the parent layer.
#[derive(Debug, Clone, Copy)]
pub struct FilterLayerPlacement {
    /// The conceptual bounding box of the pixmap that needs to be allocated to render
    /// a layer correctly, including the area affected by the filter.
    ///
    /// For example, if the filter layer contains a rect spanning (200, 200) to (300, 300)
    /// with a blur that has a radius exceeding the rectangle 40 pixels on each side, the pixmap
    /// bbox will be (160, 160) to (340, 340).
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    pixmap_bbox: RectU16,
    /// Rectangle in the parent layer's coordinate space the filtered pixmap is composited into.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    dest_bbox: RectU16,
    /// Source x offset used when sampling from the filter pixmap.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    src_x: u16,
    /// Source y offset used when sampling from the filter pixmap.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    src_y: u16,
}

impl FilterLayerPlacement {
    pub(crate) const EMPTY: Self = Self {
        pixmap_bbox: RectU16::ZERO,
        dest_bbox: RectU16::ZERO,
        src_x: 0,
        src_y: 0,
    };

    /// Place a layer whose tight content bounds are `bbox`.
    ///
    /// `filter_plan` must fit its viewport, see [`FilterData::source_viewport_size`].
    pub(crate) fn new(bbox: RectU16, filter_plan: &FilterData) -> Self {
        if bbox.is_empty() {
            return Self::EMPTY;
        }

        // Some more detailed explanations of what's going on here since this
        // part is a bit confusing.

        // `bbox` is the tight bounding box across all strips in the filter
        // layer. We now need to expand it by the filter padding to know how
        // large of a pixmap we actually need to allocate. Cover it with tiles,
        // retaining the partial final tile at the physical coordinate boundary.
        let padding = filter_plan.filter_padding;
        let pixmap_bbox = bbox
            .expand(PaddingU16::new(
                narrow(padding.left),
                narrow(padding.top),
                narrow(padding.right),
                narrow(padding.bottom),
            ))
            .snap_to_tile_coordinates();

        // Remember that in `RenderContext`, we eagerly shift everything drawn by `source_shift`
        // to conservatively ensure that everything that might be needed for the filter is in the
        // viewport area. Therefore, when compositing the filter layer back, we need to undo that
        // shift.
        let (shift_x, shift_y) = filter_plan.source_shift();
        let (shift_x, shift_y) = (narrow(shift_x), narrow(shift_y));
        assert!(
            shift_x.is_multiple_of(Tile::WIDTH) && shift_y.is_multiple_of(Tile::HEIGHT),
            "filter source shift must be tile-aligned"
        );
        // For example, if `shift_x` is 20 and `pixmap_bbox.x0` is 4,
        // shifting the pixmap back would place its left edge at -16. Since we
        // start compositing at x=0, we need to skip the first 16 pixels
        // inside the cropped pixmap (`src_x = 20 - 4`). If `pixmap_bbox.x0`
        // is already >= `shift_x`, nothing is clipped and `src_x` is 0.
        let src_x = shift_x.saturating_sub(pixmap_bbox.x0);
        let src_y = shift_y.saturating_sub(pixmap_bbox.y0);
        let dest_bbox = pixmap_bbox.relative_to_origin((shift_x, shift_y));

        assert!(
            dest_bbox.x0.is_multiple_of(Tile::WIDTH) && dest_bbox.y0.is_multiple_of(Tile::HEIGHT),
            "filter destination origin must be tile-aligned"
        );

        Self {
            pixmap_bbox,
            dest_bbox,
            src_x,
            src_y,
        }
    }

    /// Place a filtered pixmap that was rendered elsewhere and already covers
    /// exactly `dest` in the parent layer.
    ///
    /// `dest` must start on a tile boundary.
    pub fn rendered_at(dest: RectU16) -> Self {
        assert!(
            dest.x0.is_multiple_of(Tile::WIDTH) && dest.y0.is_multiple_of(Tile::HEIGHT),
            "filter destination origin must be tile-aligned"
        );
        Self {
            pixmap_bbox: dest,
            dest_bbox: dest,
            src_x: 0,
            src_y: 0,
        }
    }

    /// Return the bounds of the pixmap allocated for the filter layer.
    ///
    /// The origin is tile-aligned. The exclusive edge can cut the final tile
    /// at the boundary of the physical coordinate domain.
    pub fn pixmap_bbox(self) -> RectU16 {
        self.pixmap_bbox
    }

    /// Return the bounds where the filter layer is composited into its parent.
    ///
    /// The origin is tile-aligned. The exclusive edge can be partial after
    /// clipping the allocation to the physical coordinate domain.
    pub fn dest_bbox(self) -> RectU16 {
        self.dest_bbox
    }

    /// Return the source origin of the filter layer.
    ///
    /// The origin is guaranteed to be aligned to tile coordinates.
    pub fn src_origin(self) -> (u16, u16) {
        (self.src_x, self.src_y)
    }
}

impl Default for FilterLayerPlacement {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Narrow a padding or shift of a filter that fits its `u16` viewport.
fn narrow(value: u32) -> u16 {
    u16::try_from(value).expect("a filter that fits its viewport has u16 paddings")
}

/// Precomputed data for a filter layer.
#[derive(Debug, Clone)]
pub struct FilterData {
    /// The underlying filter.
    pub filter: Filter,
    /// The transform that was in place when the filter layer was invoked.
    pub transform: Affine,
    /// Padding that needs to be added for the area where the filter is applied.
    ///
    /// Covers [`Filter::filter_expansion`] and the device-space support of the
    /// isotropic blur that is actually applied.
    pub filter_padding: PaddingU32,
    /// Padding that needs to be added to the source region for correct filter application.
    ///
    /// Covers [`Filter::source_expansion`] and the device-space support of the
    /// isotropic blur that is actually applied.
    pub source_padding: PaddingU32,
}

impl FilterData {
    /// Create precomputed data for a filter and transform.
    ///
    /// # Panics
    ///
    /// Panics if the filter parameters or the transform are not finite, a
    /// standard deviation is negative, or a padding exceeds the `u32` domain.
    pub fn new(filter: Filter, transform: Affine) -> Self {
        validate_filter_parameters(&filter, &transform);
        let mut source_expansion = filter.source_expansion(&transform);
        let mut filter_expansion = filter.filter_expansion(&transform);
        if let Some(device_blur) = device_space_blur(&filter, &transform) {
            // The applied blur is isotropic with the transform's mean scale, so a
            // transformed user-space box can miss its support on the narrower axis.
            source_expansion = source_expansion.union(device_blur.source_expansion());
            filter_expansion = filter_expansion.union(device_blur.filter_expansion());
        }

        Self {
            filter,
            transform,
            filter_padding: snapped_padding(filter_expansion),
            source_padding: snapped_padding(source_expansion),
        }
    }

    /// Return the viewport containing all source pixels needed by this filter
    /// in a parent viewport of the given size, or `None` if it does not fit
    /// the `u16` coordinate domain.
    pub fn source_viewport_size(&self, width: u16, height: u16) -> Option<SizeU16> {
        let expanded = |size: u16, before: u32, after: u32| {
            let size = u64::from(size) + u64::from(before) + u64::from(after);
            u16::try_from(size).ok()
        };
        let padding = self.source_padding;
        Some(SizeU16::from_wh(
            expanded(width, padding.left, padding.right)?,
            expanded(height, padding.top, padding.bottom)?,
        ))
    }

    /// By how much to shift all rendered contents to ensure that all rendered contents
    /// are visible in the viewport [0, 0, width, height].
    pub fn source_shift(&self) -> (u32, u32) {
        (self.source_padding.left, self.source_padding.top)
    }

    /// Return how the filter moves its source, if it is a single offset or
    /// drop shadow primitive.
    pub fn displacement(&self) -> Option<Displacement> {
        let [primitive] = self.filter.graph.primitives.as_slice() else {
            return None;
        };
        let (dx, dy, std_deviation) = match primitive {
            FilterPrimitive::Offset { dx, dy } => {
                let (dx, dy) = transform_offset_params(*dx, *dy, &self.transform);
                (dx, dy, 0.0)
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                ..
            }
            | FilterPrimitive::DropShadowOnly {
                dx,
                dy,
                std_deviation,
                ..
            } => transform_shadow_params(*dx, *dy, *std_deviation, &self.transform),
            _ => return None,
        };
        let blur = FilterPrimitive::GaussianBlur {
            std_deviation,
            edge_mode: EdgeMode::None,
        };
        Some(Displacement {
            offset: (round(dx) as i64, round(dy) as i64),
            blur_padding: snapped_padding(blur.filter_expansion()),
            keeps_source: matches!(primitive, FilterPrimitive::DropShadow { .. }),
        })
    }
}

/// How an offset or drop shadow filter moves its source.
///
/// The output is a copy of the source moved by whole device pixels, blurred
/// and recolored first for a drop shadow, with the source itself composited
/// over it when `keeps_source` is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Displacement {
    /// The translation of the moved copy, rounded half away from zero.
    pub offset: (i64, i64),
    /// How far the blur of the moved copy reads around each pixel; zero for an offset.
    pub blur_padding: PaddingU32,
    /// Whether the unmoved source is composited over the moved copy.
    pub keeps_source: bool,
}

fn snapped_padding(expansion: Rect) -> PaddingU32 {
    fn padding(value: f64, step: u16) -> u32 {
        let snapped = snap_up(value, step);
        assert!(
            snapped <= f64::from(u32::MAX),
            "filter padding exceeds u32 coordinate domain"
        );
        snapped as u32
    }

    debug_assert!(
        expansion.x0 <= 0.0 && expansion.y0 <= 0.0 && expansion.x1 >= 0.0 && expansion.y1 >= 0.0,
        "filter expansion must contain the origin"
    );

    // TODO: We technically shouldn't need to snap here. `source_padding` is only
    // used to shift the contents when rendering into the render context, and the
    // final pixmap bbox (which is derived from `filter_expansion` will be snapped
    // separately. However, not snapping here causes larger mismatches with Vello GPU
    // since the size of the final pixmap determines in which way we decimate for the
    // gaussian blur filter. Therefore, we keep this for compatibility.
    PaddingU32::new(
        padding(-expansion.x0, Tile::WIDTH),
        padding(-expansion.y0, Tile::HEIGHT),
        padding(expansion.x1, Tile::WIDTH),
        padding(expansion.y1, Tile::HEIGHT),
    )
}

/// Return the single blur or shadow primitive of `filter` with its parameters
/// mapped to device space, as `PreparedFilter` applies it.
fn device_space_blur(filter: &Filter, transform: &Affine) -> Option<FilterPrimitive> {
    let [primitive] = filter.graph.primitives.as_slice() else {
        return None;
    };
    let mut device = primitive.clone();
    match &mut device {
        FilterPrimitive::GaussianBlur { std_deviation, .. } => {
            *std_deviation = transform_blur_params(*std_deviation, transform);
        }
        FilterPrimitive::DropShadow {
            dx,
            dy,
            std_deviation,
            ..
        }
        | FilterPrimitive::DropShadowOnly {
            dx,
            dy,
            std_deviation,
            ..
        } => {
            (*dx, *dy, *std_deviation) =
                transform_shadow_params(*dx, *dy, *std_deviation, transform);
        }
        _ => return None,
    }
    Some(device)
}

// Validate the primitives themselves: cached expansion unions can erase NaN bounds.
fn validate_filter_parameters(filter: &Filter, transform: &Affine) {
    assert!(
        transform.as_coeffs().iter().all(|value| value.is_finite()),
        "filter transform must be finite"
    );
    for primitive in &filter.graph.primitives {
        let (offset, std_deviation) = match primitive {
            FilterPrimitive::Offset { dx, dy } => (Some((*dx, *dy)), None),
            FilterPrimitive::GaussianBlur { std_deviation, .. } => (None, Some(*std_deviation)),
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                ..
            }
            | FilterPrimitive::DropShadowOnly {
                dx,
                dy,
                std_deviation,
                ..
            } => (Some((*dx, *dy)), Some(*std_deviation)),
            _ => (None, None),
        };
        if let Some((dx, dy)) = offset {
            validate_offset(dx, dy);
        }
        if let Some(std_deviation) = std_deviation {
            assert!(
                std_deviation.is_finite() && std_deviation >= 0.0,
                "filter standard deviation must be finite and non-negative"
            );
        }
    }
}

fn validate_offset(dx: f32, dy: f32) {
    assert!(
        dx.is_finite() && dy.is_finite(),
        "filter offset must be finite"
    );
}

#[cfg(feature = "std")]
fn round(value: f32) -> f32 {
    value.round()
}

#[cfg(not(feature = "std"))]
fn round(value: f32) -> f32 {
    crate::kurbo::common::FloatFuncs::round(value)
}

/// Transform an offset's dx/dy using the affine transformation's linear part.
///
/// # Returns
/// A tuple of (`scaled_dx`, `scaled_dy`) in device space.
fn transform_offset_params(dx: f32, dy: f32, transform: &Affine) -> (f32, f32) {
    let offset = Vec2::new(dx as f64, dy as f64);
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let transformed_offset = Vec2::new(a * offset.x + c * offset.y, b * offset.x + d * offset.y);
    let result = (transformed_offset.x as f32, transformed_offset.y as f32);
    validate_offset(result.0, result.1);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::palette::css::RED;

    fn offset(dx: f32, dy: f32) -> FilterData {
        FilterData::new(
            Filter::from_primitive(FilterPrimitive::Offset { dx, dy }),
            Affine::IDENTITY,
        )
    }

    fn blur(std_deviation: f32) -> Filter {
        Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation,
            edge_mode: EdgeMode::None,
        })
    }

    fn sides(padding: PaddingU32) -> (u32, u32, u32, u32) {
        (padding.left, padding.top, padding.right, padding.bottom)
    }

    #[test]
    fn paddings_beyond_u16_are_exact() {
        let positive = offset(70001.0, 70002.0);
        assert_eq!(sides(positive.source_padding), (70004, 70004, 0, 0));
        assert_eq!(sides(positive.filter_padding), (0, 0, 70004, 70004));
        assert_eq!(positive.source_shift(), (70004, 70004));
        let negative = offset(-70001.0, -70002.0);
        assert_eq!(sides(negative.source_padding), (0, 0, 70004, 70004));
        assert_eq!(sides(negative.filter_padding), (70004, 70004, 0, 0));
    }

    #[test]
    fn source_viewport_fits_only_within_u16() {
        // Blur 1 pads each side by 4 pixels.
        let data = FilterData::new(blur(1.0), Affine::IDENTITY);
        assert_eq!(
            data.source_viewport_size(65527, 4),
            Some(SizeU16::from_wh(65535, 12))
        );
        assert_eq!(data.source_viewport_size(65528, 4), None);
        assert_eq!(data.source_viewport_size(4, 65528), None);
        assert_eq!(offset(70000.0, 0.0).source_viewport_size(4, 4), None);
    }

    #[test]
    fn offset_displacement_rounds_half_away_from_zero_in_device_space() {
        for (amount, rounded) in [(-4.5, -5), (-0.5, -1), (0.5, 1), (4.5, 5)] {
            assert_eq!(
                offset(amount, -amount).displacement(),
                Some(Displacement {
                    offset: (rounded, -rounded),
                    blur_padding: PaddingU32::ZERO,
                    keeps_source: false,
                })
            );
        }
        let scaled = FilterData::new(
            Filter::from_primitive(FilterPrimitive::Offset { dx: 1.5, dy: 0.0 }),
            Affine::scale(2.0),
        );
        assert_eq!(scaled.displacement().unwrap().offset, (3, 0));
        assert_eq!(
            FilterData::new(blur(1.0), Affine::IDENTITY).displacement(),
            None
        );
    }

    #[test]
    fn shadow_displacement_pads_only_the_applied_blur() {
        for (primitive, keeps_source) in [
            (
                FilterPrimitive::DropShadow {
                    dx: 1.5,
                    dy: -70_000.0,
                    std_deviation: 8.0,
                    edge_mode: EdgeMode::None,
                    color: RED,
                },
                true,
            ),
            (
                FilterPrimitive::DropShadowOnly {
                    dx: 1.5,
                    dy: -70_000.0,
                    std_deviation: 8.0,
                    edge_mode: EdgeMode::None,
                    color: RED,
                },
                false,
            ),
        ] {
            let data = FilterData::new(Filter::from_primitive(primitive), Affine::scale(2.0));
            assert_eq!(
                data.displacement(),
                Some(Displacement {
                    offset: (3, -140_000),
                    blur_padding: PaddingU32::new(48, 48, 48, 48),
                    keeps_source,
                })
            );
        }
    }

    #[test]
    #[should_panic(expected = "filter padding exceeds u32 coordinate domain")]
    fn rejects_padding_beyond_u32() {
        offset(u32::MAX as f32, 0.0);
    }

    #[test]
    #[should_panic(expected = "filter offset must be finite")]
    fn rejects_nan_offset_hidden_by_expansion_union() {
        offset(f32::NAN, 0.0);
    }

    #[test]
    #[should_panic(expected = "filter offset must be finite")]
    fn rejects_offset_that_overflows_in_device_space() {
        PreparedFilter::new(
            &Filter::from_primitive(FilterPrimitive::Offset { dx: 2.0, dy: 0.0 }),
            &Affine::scale(f64::MAX),
        );
    }

    #[test]
    #[should_panic(expected = "filter transform must be finite")]
    fn rejects_nonfinite_transform() {
        FilterData::new(blur(1.0), Affine::scale(f64::INFINITY));
    }

    #[test]
    fn rejects_nonfinite_or_negative_standard_deviation() {
        for std_deviation in [f32::NAN, f32::INFINITY, -1.0] {
            let filter = blur(std_deviation);
            let data =
                std::panic::catch_unwind(|| FilterData::new(filter.clone(), Affine::IDENTITY));
            assert!(data.is_err(), "{std_deviation}");
            let prepared =
                std::panic::catch_unwind(|| PreparedFilter::new(&filter, &Affine::IDENTITY));
            assert!(prepared.is_err(), "{std_deviation}");
        }
    }

    #[test]
    fn blur_padding_covers_the_isotropic_blur_actually_applied() {
        for (transform, horizontal, vertical) in [
            (Affine::scale_non_uniform(0.0, 2.0), 24, 48),
            (Affine::scale_non_uniform(2.0, 3.0), 60, 72),
            (Affine::new([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]), 48, 28),
        ] {
            let data = FilterData::new(blur(8.0), transform);
            for padding in [data.source_padding, data.filter_padding] {
                assert_eq!(
                    sides(padding),
                    (horizontal, vertical, horizontal, vertical),
                    "{transform:?}"
                );
            }
        }
    }

    #[test]
    fn shadow_padding_covers_offset_and_applied_blur() {
        for primitive in [
            FilterPrimitive::DropShadow {
                dx: 5.0,
                dy: -7.0,
                std_deviation: 8.0,
                edge_mode: EdgeMode::None,
                color: RED,
            },
            FilterPrimitive::DropShadowOnly {
                dx: 5.0,
                dy: -7.0,
                std_deviation: 8.0,
                edge_mode: EdgeMode::None,
                color: RED,
            },
        ] {
            let data = FilterData::new(
                Filter::from_primitive(primitive),
                Affine::scale_non_uniform(0.0, 2.0),
            );
            assert_eq!(sides(data.source_padding), (24, 36, 24, 64));
            assert_eq!(sides(data.filter_padding), (24, 64, 24, 36));
        }
    }
}
