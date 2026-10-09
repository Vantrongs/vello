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
use crate::filter_effects::{Filter, FilterPrimitive};
use crate::geometry::{PaddingU32, RectU32, SizeU32};
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
    /// No pixel operation: the layer placement already implements the filter.
    Identity,
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
    pixmap_bbox: RectU32,
    /// Rectangle in the parent layer's coordinate space the filtered pixmap is composited into.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    dest_bbox: RectU32,
    /// Source x offset used when sampling from the filter pixmap.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    src_x: u32,
    /// Source y offset used when sampling from the filter pixmap.
    ///
    /// See the comments in `FilterLayerPlacement::new` for more information.
    src_y: u32,
}

impl FilterLayerPlacement {
    pub(crate) const EMPTY: Self = Self {
        pixmap_bbox: RectU32::ZERO,
        dest_bbox: RectU32::ZERO,
        src_x: 0,
        src_y: 0,
    };

    pub(crate) fn new(bbox: RectU32, filter_plan: &FilterData) -> Self {
        if bbox.is_empty() {
            return Self::EMPTY;
        }

        // Some more detailed explanations of what's going on here since this
        // part is a bit confusing.

        // `bbox` is the tight bounding box across all strips in the filter
        // layer. We now need to expand it by the filter padding to know how
        // large of a pixmap we actually need to allocate. Cover it with tiles,
        // without clipping the source to the physical target dimensions.
        let pixmap_bbox = bbox
            .expand(filter_plan.filter_padding)
            .snap_to_tile_coordinates();

        // Remember that in `RenderContext`, we eagerly shift everything drawn by `source_shift`
        // to conservatively ensure that everything that might be needed for the filter is in the
        // viewport area. Therefore, when compositing the filter layer back, we need to undo that
        // shift.
        let (shift_x, shift_y) = filter_plan.source_shift();
        assert!(
            shift_x.is_multiple_of(Tile::WIDTH_U32) && shift_y.is_multiple_of(Tile::HEIGHT_U32),
            "filter source shift must be tile-aligned"
        );
        if let Some((dx, dy)) = filter_plan.placement_offset {
            let translation = (dx - i64::from(shift_x), dy - i64::from(shift_y));
            let dest_bbox = pixmap_bbox.shift(translation);
            if dest_bbox.is_empty() {
                return Self::EMPTY;
            }
            let source_origin = |origin: u32, shift: i64| {
                u32::try_from((-(i64::from(origin) + shift)).max(0))
                    .expect("filter source origin exceeds u32 coordinate domain")
            };
            return Self {
                pixmap_bbox,
                dest_bbox,
                src_x: source_origin(pixmap_bbox.x0, translation.0),
                src_y: source_origin(pixmap_bbox.y0, translation.1),
            };
        }

        // For example, if `shift_x` is 20 and `pixmap_bbox.x0` is 4,
        // shifting the pixmap back would place its left edge at -16. Since we
        // start compositing at x=0, we need to skip the first 16 pixels
        // inside the cropped pixmap (`src_x = 20 - 4`). If `pixmap_bbox.x0`
        // is already >= `shift_x`, nothing is clipped and `src_x` is 0.
        let src_x = shift_x.saturating_sub(pixmap_bbox.x0);
        let src_y = shift_y.saturating_sub(pixmap_bbox.y0);
        let dest_bbox = pixmap_bbox.relative_to_origin((shift_x, shift_y));

        assert!(
            dest_bbox.x0.is_multiple_of(Tile::WIDTH_U32)
                && dest_bbox.y0.is_multiple_of(Tile::HEIGHT_U32),
            "filter destination origin must be tile-aligned"
        );

        Self {
            pixmap_bbox,
            dest_bbox,
            src_x,
            src_y,
        }
    }

    /// Return the bounds of the pixmap allocated for the filter layer.
    ///
    /// All edges are tile-aligned in the source coordinate domain.
    pub fn pixmap_bbox(self) -> RectU32 {
        self.pixmap_bbox
    }

    /// Return the bounds where the filter layer is composited into its parent.
    ///
    /// Offset filters can produce edges between tile boundaries.
    pub fn dest_bbox(self) -> RectU32 {
        self.dest_bbox
    }

    /// Return the source origin of the filter layer.
    ///
    /// The origin includes any pixels clipped at the parent viewport's left/top edge.
    pub fn src_origin(self) -> (u32, u32) {
        (self.src_x, self.src_y)
    }
}

impl Default for FilterLayerPlacement {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Precomputed data for a filter layer.
#[derive(Debug, Clone)]
pub struct FilterData {
    /// The underlying filter.
    pub filter: Filter,
    /// The transform that was in place when the filter layer was invoked.
    pub transform: Affine,
    /// Padding that needs to be added to the intermediate filter image.
    ///
    /// Includes [`Filter::filter_expansion`] and the device-space support of the
    /// implemented isotropic blur. A pure offset is represented by layer placement
    /// and does not allocate its empty translation gap.
    pub filter_padding: PaddingU32,
    /// Padding that needs to be added to the source region for correct filter application.
    ///
    /// See [`Filter::source_expansion`].
    pub source_padding: PaddingU32,
    /// A pure offset is represented by placement, avoiding an allocation across its empty gap.
    placement_offset: Option<(i64, i64)>,
}

impl FilterData {
    /// Return the viewport containing all source pixels needed by this filter.
    ///
    /// # Panics
    ///
    /// Panics if the expanded viewport exceeds the source coordinate domain.
    /// Truncating it would silently discard source pixels that affect the output.
    pub fn source_viewport_size(&self, width: u32, height: u32) -> SizeU32 {
        let expanded = |size: u32, before: u32, after: u32| {
            size.checked_add(before)
                .and_then(|size| size.checked_add(after))
                .filter(|&size| size.checked_next_multiple_of(Tile::WIDTH_U32).is_some())
                .expect("filter source viewport exceeds u32 coordinate domain")
        };
        SizeU32::from_wh(
            expanded(width, self.source_padding.left, self.source_padding.right),
            expanded(height, self.source_padding.top, self.source_padding.bottom),
        )
    }

    /// Create precomputed data for a filter and transform.
    pub fn new(filter: Filter, transform: Affine) -> Self {
        validate_filter_parameters(&filter, &transform);
        fn snapped_padding(expansion: Rect) -> PaddingU32 {
            fn padding(value: f64, step: u16) -> u32 {
                let snapped = snap_up(value, step);
                assert!(
                    snapped.is_finite() && snapped >= 0.0 && snapped <= f64::from(u32::MAX),
                    "filter padding exceeds u32 coordinate domain"
                );
                snapped as u32
            }
            assert!(
                expansion.x0 <= 0.0
                    && expansion.y0 <= 0.0
                    && expansion.x1 >= 0.0
                    && expansion.y1 >= 0.0,
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

        let mut source_expansion = filter.source_expansion(&transform);
        let mut filter_expansion = filter.filter_expansion(&transform);
        if let [primitive] = filter.graph.primitives.as_slice() {
            let mut device_blur = primitive.clone();
            let is_blur = match &mut device_blur {
                FilterPrimitive::GaussianBlur { std_deviation, .. } => {
                    *std_deviation = transform_blur_params(*std_deviation, &transform);
                    true
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
                        transform_shadow_params(*dx, *dy, *std_deviation, &transform);
                    true
                }
                _ => false,
            };
            if is_blur {
                // The implemented blur is isotropic after averaging the transform's
                // singular values. A transformed user-space box alone can omit
                // support on the narrower axis, especially for singular transforms.
                source_expansion = source_expansion.union(device_blur.source_expansion());
                filter_expansion = filter_expansion.union(device_blur.filter_expansion());
            }
        }
        let source_padding = snapped_padding(source_expansion);
        let mut filter_padding = snapped_padding(filter_expansion);
        let placement_offset = match filter.graph.primitives.as_slice() {
            [FilterPrimitive::Offset { dx, dy }] => {
                let (dx, dy) = transform_offset_params(*dx, *dy, &transform);
                filter_padding = PaddingU32::ZERO;
                #[cfg(feature = "std")]
                let rounded = (dx.round() as i64, dy.round() as i64);
                #[cfg(not(feature = "std"))]
                let rounded = (
                    crate::kurbo::common::FloatFuncs::round(dx) as i64,
                    crate::kurbo::common::FloatFuncs::round(dy) as i64,
                );
                Some(rounded)
            }
            _ => None,
        };

        Self {
            filter,
            transform,
            filter_padding,
            source_padding,
            placement_offset,
        }
    }

    /// Prepare the pixel operation remaining after this layer's placement is applied.
    pub fn prepare_for_layer(&self) -> PreparedFilter {
        if self.placement_offset.is_some() {
            PreparedFilter::Identity
        } else {
            PreparedFilter::new(&self.filter, &self.transform)
        }
    }

    /// By how much to shift all rendered contents to ensure that all rendered contents
    /// are visible in the viewport [0, 0, width, height].
    pub fn source_shift(&self) -> (u32, u32) {
        (self.source_padding.left, self.source_padding.top)
    }
}

// Validate the primitives themselves: cached expansion unions can erase NaN bounds.
fn validate_filter_parameters(filter: &Filter, transform: &Affine) {
    assert!(
        transform.as_coeffs().iter().all(|value| value.is_finite()),
        "filter transform must be finite"
    );
    for primitive in &filter.graph.primitives {
        let (offset, sigma) = match primitive {
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
        if let Some(sigma) = sigma {
            assert!(
                sigma.is_finite() && sigma >= 0.0,
                "filter standard deviation must be finite and non-negative"
            );
        }
    }
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

fn validate_offset(dx: f32, dy: f32) {
    assert!(
        dx.is_finite() && dy.is_finite(),
        "filter offset must be finite"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offset(dx: f32, dy: f32) -> FilterData {
        FilterData::new(
            Filter::from_primitive(FilterPrimitive::Offset { dx, dy }),
            Affine::IDENTITY,
        )
    }

    #[test]
    fn padding_and_source_viewports_cross_physical_limits_in_both_directions() {
        let positive = offset(70001.0, 70002.0);
        assert_eq!(positive.source_padding, PaddingU32::new(70004, 70004, 0, 0));
        assert_eq!(positive.filter_padding, PaddingU32::ZERO);
        assert_eq!(
            positive.source_viewport_size(65535, 65535),
            SizeU32::new(135539)
        );
        let negative = offset(-70001.0, -70002.0);
        assert_eq!(negative.source_padding, PaddingU32::new(0, 0, 70004, 70004));
        assert_eq!(negative.filter_padding, PaddingU32::ZERO);
        assert_eq!(
            negative.source_viewport_size(65535, 65535),
            SizeU32::new(135539)
        );
    }

    #[test]
    fn placement_retains_large_source_origin_before_rebasing() {
        let data = offset(70000.0, 0.0);
        let placement = FilterLayerPlacement::new(RectU32::new(70004, 4, 70008, 8), &data);
        assert_eq!(placement.pixmap_bbox(), RectU32::new(70004, 4, 70008, 8));
        assert_eq!(placement.dest_bbox(), RectU32::new(70004, 4, 70008, 8));
        assert_eq!(placement.src_origin(), (0, 0));
    }

    #[test]
    fn pure_offset_rebases_a_small_source_without_allocating_the_empty_gap() {
        let data = offset(-70000.0, 0.0);
        let placement = FilterLayerPlacement::new(RectU32::new(70004, 4, 70008, 8), &data);
        assert_eq!(placement.pixmap_bbox(), RectU32::new(70004, 4, 70008, 8));
        assert_eq!(placement.dest_bbox(), RectU32::new(4, 4, 8, 8));
        assert_eq!(placement.src_origin(), (0, 0));
        assert!(matches!(data.prepare_for_layer(), PreparedFilter::Identity));
        assert!(matches!(
            PreparedFilter::new(&data.filter, &data.transform),
            PreparedFilter::Offset(_)
        ));
    }

    #[test]
    fn offset_placement_rounds_half_away_from_zero_and_clips_exact_pixels() {
        for (amount, rounded) in [(-4.5, -5_i64), (-0.5, -1), (0.5, 1), (4.5, 5)] {
            let data = offset(amount, amount);
            let (shift_x, shift_y) = data.source_shift();
            let bbox = RectU32::new(shift_x, shift_y, shift_x + 8, shift_y + 8);
            let placement = FilterLayerPlacement::new(bbox, &data);
            assert_eq!(placement.pixmap_bbox(), bbox);
            assert_eq!(
                placement.dest_bbox(),
                RectU32::new(
                    rounded.max(0) as u32,
                    rounded.max(0) as u32,
                    (8 + rounded) as u32,
                    (8 + rounded) as u32
                )
            );
            let crop = (-rounded).max(0) as u32;
            assert_eq!(placement.src_origin(), (crop, crop));
        }
        let empty = FilterLayerPlacement::new(RectU32::new(0, 0, 4, 4), &offset(-8.0, 0.0));
        assert!(empty.dest_bbox().is_empty());
        assert!(empty.pixmap_bbox().is_empty());
    }

    #[test]
    #[should_panic(expected = "filter padding exceeds u32 coordinate domain")]
    fn rejects_unrepresentable_padding_before_float_conversion() {
        offset(u32::MAX as f32, 0.0);
    }

    #[test]
    #[should_panic(expected = "filter source viewport exceeds u32 coordinate domain")]
    fn rejects_source_viewport_addition_overflow() {
        offset(4.0, 0.0).source_viewport_size(u32::MAX - 4, 1);
    }

    #[test]
    #[should_panic(expected = "filter offset must be finite")]
    fn rejects_nan_even_when_expansion_union_erases_it() {
        offset(f32::NAN, 0.0);
    }

    #[test]
    #[should_panic(expected = "filter offset must be finite")]
    fn prepared_filter_rejects_nonfinite_offset() {
        PreparedFilter::new(
            &Filter::from_primitive(FilterPrimitive::Offset {
                dx: 0.0,
                dy: f32::INFINITY,
            }),
            &Affine::IDENTITY,
        );
    }

    #[test]
    #[should_panic(expected = "filter offset must be finite")]
    fn prepared_filter_rejects_transform_overflow() {
        PreparedFilter::new(
            &Filter::from_primitive(FilterPrimitive::Offset { dx: 2.0, dy: 0.0 }),
            &Affine::scale(f64::MAX),
        );
    }
    #[test]
    #[should_panic(expected = "filter transform must be finite")]
    fn rejects_nonfinite_transform() {
        FilterData::new(
            Filter::from_primitive(FilterPrimitive::Offset { dx: 1.0, dy: 0.0 }),
            Affine::scale(f64::INFINITY),
        );
    }

    #[test]
    fn transformed_blur_padding_covers_actual_isotropic_support() {
        use crate::filter_effects::EdgeMode;
        for (transform, horizontal, vertical) in [
            (Affine::scale_non_uniform(0.0, 2.0), 24, 48),
            (Affine::scale_non_uniform(2.0, 3.0), 60, 72),
            (Affine::new([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]), 48, 28),
        ] {
            let data = FilterData::new(
                Filter::from_primitive(FilterPrimitive::GaussianBlur {
                    std_deviation: 8.0,
                    edge_mode: EdgeMode::None,
                }),
                transform,
            );
            for padding in [data.source_padding, data.filter_padding] {
                assert_eq!(
                    (padding.left, padding.top, padding.right, padding.bottom),
                    (horizontal, vertical, horizontal, vertical),
                    "{transform:?}"
                );
            }
        }
    }

    #[test]
    fn shadow_padding_covers_transformed_offset_and_actual_blur_in_both_directions() {
        use crate::color::palette::css::RED;
        use crate::filter_effects::EdgeMode;
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
            let source = data.source_padding;
            let dest = data.filter_padding;
            assert_eq!(
                (source.left, source.top, source.right, source.bottom),
                (24, 36, 24, 64)
            );
            assert_eq!(
                (dest.left, dest.top, dest.right, dest.bottom),
                (24, 64, 24, 36)
            );
        }
    }

    #[test]
    fn nonfinite_blur_and_transform_are_rejected_before_preparation() {
        use crate::filter_effects::EdgeMode;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let invalid_blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: invalid,
                edge_mode: EdgeMode::None,
            });
            assert!(
                std::panic::catch_unwind(|| FilterData::new(
                    invalid_blur.clone(),
                    Affine::IDENTITY
                ))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| PreparedFilter::new(&invalid_blur, &Affine::IDENTITY))
                    .is_err()
            );
            let blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: 1.0,
                edge_mode: EdgeMode::None,
            });
            let transform = Affine::scale(f64::from(invalid));
            assert!(std::panic::catch_unwind(|| FilterData::new(blur.clone(), transform)).is_err());
            assert!(std::panic::catch_unwind(|| PreparedFilter::new(&blur, &transform)).is_err());
        }
    }
}
