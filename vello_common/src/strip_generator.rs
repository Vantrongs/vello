// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::clip::{ClipRef, ClipShape, PathDataRef, intersect};
use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::geometry::RectU32;
use crate::kurbo::{Affine, PathEl, Rect, Stroke};
use crate::peniko::Fill;
use crate::strip::{Strip, StripFillSegment, visit_strip_fill_segments};
use crate::tile::{Tile, Tiles};
use crate::util::strip_bbox;
use crate::{flatten, rect, strip};
use alloc::vec::Vec;
use peniko::kurbo::StrokeCtx;

/// A storage for storing strip-related data.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StripStorage {
    /// The strips in the storage.
    pub strips: Vec<Strip>,
    /// The alphas in the storage.
    pub alphas: Vec<u8>,
    generation_mode: GenerationMode,
}

/// The generation mode of the strip storage.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum GenerationMode {
    #[default]
    /// Clear strips before generating the new ones.
    Replace,
    /// Don't clear strips, append to the existing buffer.
    Append,
    /// Truncate strips to the given index before generating new ones,
    /// preserving strips in `[0..n]`.
    ReplaceAfter(usize),
}

impl StripStorage {
    /// Create a new strip storage with the given generation mode.
    pub fn new(generation_mode: GenerationMode) -> Self {
        Self {
            strips: Vec::new(),
            alphas: Vec::new(),
            generation_mode,
        }
    }

    /// Reset the storage.
    pub fn clear(&mut self) {
        self.strips.clear();
        self.alphas.clear();
    }

    /// Get the current generation mode.
    pub fn generation_mode(&self) -> GenerationMode {
        self.generation_mode
    }

    /// Set the generation mode of the storage.
    pub fn set_generation_mode(&mut self, mode: GenerationMode) {
        self.generation_mode = mode;
    }

    /// Whether the strip storage is empty.
    pub fn is_empty(&self) -> bool {
        self.strips.is_empty() && self.alphas.is_empty()
    }

    /// Extend the current strip storage with the data from another storage.
    pub fn extend(&mut self, other: &Self) {
        self.strips.extend(&other.strips);
        self.alphas.extend(&other.alphas);
    }
}

/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    pub(crate) level: Level,
    line_buf: Vec<Line>,
    source_path: Vec<PathEl>,
    source_storage: StripStorage,
    source_segments: Vec<SourceSegment>,
    #[cfg(test)]
    source_window_replays: usize,
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    temp_storage: StripStorage,
    tiles: Tiles,
    width: u32,
    height: u32,
}

// Windows only partition geometry generation; filter images and filter phases remain whole.
const SOURCE_WINDOW: u32 = 65532;

// Restrict iteration to intersecting windows without changing the source grid's phase.
fn source_window_grid(bounds: RectU32, cull_bbox: RectU32) -> RectU32 {
    let visible = bounds.intersect(cull_bbox);
    if visible.is_empty() {
        return RectU32::ZERO;
    }
    let start = |origin: u32, first: u32| origin + (first - origin) / SOURCE_WINDOW * SOURCE_WINDOW;
    RectU32::new(
        start(bounds.x0, visible.x0),
        start(bounds.y0, visible.y0),
        visible.x1,
        visible.y1,
    )
}

#[derive(Debug)]
struct SourceSegment {
    fill: StripFillSegment,
    alpha_idx: Option<u32>,
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u32, height: u32, level: Level) -> Self {
        Self {
            level,
            line_buf: Vec::new(),
            source_path: Vec::new(),
            source_storage: StripStorage::default(),
            source_segments: Vec::new(),
            #[cfg(test)]
            source_window_replays: 0,
            tiles: Tiles::new(level, 0, 0),
            flatten_ctx: FlattenCtx::default(),
            stroke_ctx: StrokeCtx::default(),
            temp_storage: StripStorage::default(),
            width,
            height,
        }
    }

    /// Get this strip generator's viewport width.
    #[inline(always)]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get this strip generator's viewport height.
    #[inline(always)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Generate the strips for a filled path.
    pub fn generate_filled_path(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        let cull_bbox = clip_path
            .map(|clip_path| clip_path.bbox)
            .unwrap_or(RectU32::new(0, 0, self.width, self.height));
        let mut path = path.into_iter();
        let bounds = self.source_bounds(&mut path, transform, None);
        if let Some(bounds) = bounds
            .filter(|bounds| bounds.width() > SOURCE_WINDOW || bounds.height() > SOURCE_WINDOW)
        {
            self.generate_source_windows(
                bounds,
                cull_bbox,
                transform,
                None,
                fill_rule,
                aliasing_threshold,
                strip_storage,
                clip_path,
            );
            self.source_path.clear();
            return;
        }
        let origin = bounds.map_or((0, 0), |bounds| (bounds.x0, bounds.y0));
        let local_transform = if origin == (0, 0) {
            transform
        } else {
            Affine::translate((-f64::from(origin.0), -f64::from(origin.1))) * transform
        };
        flatten::fill(
            self.level,
            self.source_path.iter().copied().chain(path),
            local_transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
            if origin == (0, 0) {
                cull_bbox
            } else {
                cull_bbox.relative_to_origin(origin)
            },
        );
        self.source_path.clear();

        self.generate_with_clip(
            aliasing_threshold,
            strip_storage,
            fill_rule,
            clip_path,
            origin,
            bounds.map_or((self.width, self.height), |bounds| {
                (bounds.width(), bounds.height())
            }),
        );
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        let cull_bbox = clip_path
            .map(|clip_path| clip_path.bbox)
            .unwrap_or(RectU32::new(0, 0, self.width, self.height));
        let mut path = path.into_iter();
        let bounds = self.source_bounds(&mut path, transform, Some(stroke));
        if let Some(bounds) = bounds
            .filter(|bounds| bounds.width() > SOURCE_WINDOW || bounds.height() > SOURCE_WINDOW)
        {
            self.generate_source_windows(
                bounds,
                cull_bbox,
                transform,
                Some(stroke),
                Fill::NonZero,
                aliasing_threshold,
                strip_storage,
                clip_path,
            );
            self.source_path.clear();
            return;
        }
        let origin = bounds.map_or((0, 0), |bounds| (bounds.x0, bounds.y0));
        let local_transform = if origin == (0, 0) {
            transform
        } else {
            Affine::translate((-f64::from(origin.0), -f64::from(origin.1))) * transform
        };
        flatten::stroke(
            self.level,
            self.source_path.iter().copied().chain(path),
            stroke,
            local_transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
            &mut self.stroke_ctx,
            if origin == (0, 0) {
                cull_bbox
            } else {
                cull_bbox.relative_to_origin(origin)
            },
        );
        self.source_path.clear();
        self.generate_with_clip(
            aliasing_threshold,
            strip_storage,
            Fill::NonZero,
            clip_path,
            origin,
            bounds.map_or((self.width, self.height), |bounds| {
                (bounds.width(), bounds.height())
            }),
        );
    }

    fn source_bounds(
        &mut self,
        path: &mut impl Iterator<Item = PathEl>,
        transform: Affine,
        stroke: Option<&Stroke>,
    ) -> Option<RectU32> {
        #[cfg(test)]
        {
            self.source_window_replays = 0;
        }
        if self.width <= u32::from(u16::MAX) && self.height <= u32::from(u16::MAX) {
            return None;
        }
        self.source_path.clear();
        self.source_path.extend(path);
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut finite = true;
        let mut include = |point: crate::kurbo::Point| {
            finite &= point.x.is_finite() && point.y.is_finite();
            bounds[0] = bounds[0].min(point.x);
            bounds[1] = bounds[1].min(point.y);
            bounds[2] = bounds[2].max(point.x);
            bounds[3] = bounds[3].max(point.y);
        };
        for element in &self.source_path {
            match transform * *element {
                PathEl::MoveTo(p) | PathEl::LineTo(p) => include(p),
                PathEl::QuadTo(a, b) => {
                    include(a);
                    include(b);
                }
                PathEl::CurveTo(a, b, c) => {
                    include(a);
                    include(b);
                    include(c);
                }
                PathEl::ClosePath => {}
            }
        }
        if let Some(stroke) = stroke {
            let expansion =
                flatten::stroke_cull_rect(RectU32::ZERO, stroke, flatten::max_scale(transform));
            for i in 0..4 {
                bounds[i] += expansion[i];
            }
        }
        if !finite || bounds.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let minimum =
            |value: f64, size: u32, tile: u32| (value.max(0.0) as u32).min(size) / tile * tile;
        // Retain the boundary tile, including zero coverage: it is part of the
        // existing tight filter-image bounds and therefore of blur decimation phase.
        let maximum = |value: f64, size: u32, tile: u32| {
            (value.max(0.0) + f64::from(tile)).min(f64::from(size)) as u32
        };
        let x0 = minimum(bounds[0], self.width, Tile::WIDTH_U32);
        let y0 = minimum(bounds[1], self.height, Tile::HEIGHT_U32);
        Some(RectU32::new(
            x0,
            y0,
            maximum(bounds[2], self.width, Tile::WIDTH_U32).max(x0),
            maximum(bounds[3], self.height, Tile::HEIGHT_U32).max(y0),
        ))
    }

    fn generate_source_windows(
        &mut self,
        bounds: RectU32,
        cull_bbox: RectU32,
        transform: Affine,
        stroke: Option<&Stroke>,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        self.source_storage.clear();
        self.source_segments.clear();
        let grid = source_window_grid(bounds, cull_bbox);
        let mut y = grid.y0;
        while y < grid.y1 {
            let y1 = y.saturating_add(SOURCE_WINDOW).min(bounds.y1);
            let mut x = grid.x0;
            while x < grid.x1 {
                let x1 = x.saturating_add(SOURCE_WINDOW).min(bounds.x1);
                let window = RectU32::new(x, y, x1, y1);
                let local_cull = cull_bbox.intersect(window).relative_to_origin((x, y));
                if !local_cull.is_empty() {
                    #[cfg(test)]
                    {
                        self.source_window_replays += 1;
                    }
                    let local_transform =
                        Affine::translate((-f64::from(x), -f64::from(y))) * transform;
                    if let Some(stroke) = stroke {
                        flatten::stroke(
                            self.level,
                            self.source_path.iter().copied(),
                            stroke,
                            local_transform,
                            &mut self.line_buf,
                            &mut self.flatten_ctx,
                            &mut self.stroke_ctx,
                            local_cull,
                        );
                    } else {
                        flatten::fill(
                            self.level,
                            self.source_path.iter().copied(),
                            local_transform,
                            &mut self.line_buf,
                            &mut self.flatten_ctx,
                            local_cull,
                        );
                    }
                    self.tiles
                        .make_tiles_analytic_aa(self.level, &self.line_buf, x1 - x, y1 - y);
                    self.tiles.sort_tiles();
                    self.source_storage.strips.clear();
                    strip::render(
                        self.level,
                        &self.tiles,
                        &mut self.source_storage.strips,
                        &mut self.source_storage.alphas,
                        fill_rule,
                        aliasing_threshold,
                        &self.line_buf,
                    );
                    let global = |mut fill: StripFillSegment| {
                        fill.tile_x0 += x / Tile::WIDTH_U32;
                        fill.tile_x1 += x / Tile::WIDTH_U32;
                        fill.tile_y += y / Tile::HEIGHT_U32;
                        fill
                    };
                    visit_strip_fill_segments(
                        &self.source_storage.strips,
                        RectU32::new(0, 0, x1 - x, y1 - y).to_tile_bounds(),
                        &mut self.source_segments,
                        |segments, alpha| {
                            segments.push(SourceSegment {
                                fill: global(alpha.fill),
                                alpha_idx: Some(alpha.alpha_idx),
                            });
                        },
                        |segments, fill| {
                            segments.push(SourceSegment {
                                fill: global(fill),
                                alpha_idx: None,
                            });
                        },
                    );
                }
                x = x1;
            }
            y = y1;
        }
        self.source_segments
            .sort_unstable_by_key(|segment| (segment.fill.tile_y, segment.fill.tile_x0));
        render_with_clip(
            self.level,
            &mut self.temp_storage,
            strip_storage,
            clip_path,
            |strips, alphas| {
                for segment in &self.source_segments {
                    let fill = segment.fill;
                    let index = Strip::alpha_index(alphas.len());
                    strips.push(Strip::new(fill.x0(), fill.y(), index, false));
                    if let Some(start) = segment.alpha_idx {
                        let len = (fill.x1() - fill.x0()) as usize * Tile::HEIGHT_U32 as usize;
                        alphas.extend_from_slice(
                            &self.source_storage.alphas[start as usize..start as usize + len],
                        );
                    } else {
                        alphas.extend([255; 16]);
                        if fill.tile_x1 - fill.tile_x0 > 1 {
                            strips.push(Strip::new(
                                fill.x1() - Tile::WIDTH_U32,
                                fill.y(),
                                Strip::alpha_index(alphas.len()),
                                true,
                            ));
                            alphas.extend([255; 16]);
                        }
                    }
                }
                if let Some(last) = self.source_segments.last() {
                    strips.push(Strip::sentinel(
                        last.fill.y(),
                        Strip::alpha_index(alphas.len()),
                    ));
                }
            },
        );
        self.source_segments.clear();
        self.source_storage.clear();
    }

    fn generate_with_clip(
        &mut self,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        fill_rule: Fill,
        clip_path: Option<PathDataRef<'_>>,
        origin: (u32, u32),
        size: (u32, u32),
    ) {
        if self.line_buf.is_empty() {
            render_with_clip(
                self.level,
                &mut self.temp_storage,
                strip_storage,
                clip_path,
                |_, _| {},
            );
            return;
        }
        self.tiles
            .make_tiles_analytic_aa(self.level, &self.line_buf, size.0, size.1);

        self.tiles.sort_tiles();

        let level = self.level;
        let tiles = &self.tiles;
        let line_buf = &self.line_buf;
        render_with_clip(
            level,
            &mut self.temp_storage,
            strip_storage,
            clip_path,
            |strips, alphas| {
                let start = strips.len();
                strip::render(
                    level,
                    tiles,
                    strips,
                    alphas,
                    fill_rule,
                    aliasing_threshold,
                    line_buf,
                );
                for strip in &mut strips[start..] {
                    if !strip.is_sentinel() {
                        strip.x = strip
                            .x
                            .checked_add(origin.0)
                            .expect("source strip coordinate overflow");
                    }
                    strip.y = strip
                        .y
                        .checked_add(origin.1)
                        .expect("source strip coordinate overflow");
                }
            },
        );
    }

    /// Generate strips directly for a pixel-aligned rectangle.
    ///
    /// This bypasses the full path processing pipeline (flatten -> tiles -> strips)
    /// by directly creating strip coverage data for the rectangle.
    pub fn generate_filled_rect_fast(
        &mut self,
        rect: &Rect,
        strip_storage: &mut StripStorage,
        clip_path: Option<ClipRef<'_>>,
    ) -> ClipShape {
        let viewport = Rect::new(0.0, 0.0, self.width as f64, self.height as f64);
        let rect = rect.abs();
        let (clamped, complex_clip, shape) = match clip_path {
            None => {
                let clamped = rect.intersect(viewport);
                (clamped, None, ClipShape::AxisAlignedRect(clamped))
            }
            Some(ClipRef {
                shape: ClipShape::AxisAlignedRect(clip_rect),
                ..
            }) => {
                let clamped = rect.intersect(clip_rect);
                (clamped, None, ClipShape::AxisAlignedRect(clamped))
            }
            Some(ClipRef {
                path,
                shape: ClipShape::Path,
            }) => {
                // Clip bbox is always guaranteed to be within viewport bounds, so no need to
                // intersect again.
                let bbox = path.bbox;
                let clip_bbox = Rect::new(
                    f64::from(bbox.x0),
                    f64::from(bbox.y0),
                    f64::from(bbox.x1),
                    f64::from(bbox.y1),
                );
                (rect.intersect(clip_bbox), Some(path), ClipShape::Path)
            }
        };

        let level = self.level;
        render_with_clip(
            level,
            &mut self.temp_storage,
            strip_storage,
            complex_clip,
            |strips, alphas| {
                rect::render(level, clamped, strips, alphas);
            },
        );

        shape
    }

    /// Reset the strip generator for a viewport size, resizing only when needed.
    pub fn reset(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.line_buf.clear();
        self.tiles.reset(0, 0);
        self.temp_storage.clear();
    }
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static CLIP_BYPASSES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Render strips via `render_fn` with optional clip intersection.
///
/// When `clip_path` is `Some`, strips are rendered into `temp_storage` first, then
/// intersected with the clip mask into `strip_storage`, unless all their coverage is
/// within a proven opaque part of the mask. Otherwise strips are rendered directly
/// into `strip_storage`.
fn render_with_clip(
    level: Level,
    temp_storage: &mut StripStorage,
    strip_storage: &mut StripStorage,
    clip_path: Option<PathDataRef<'_>>,
    render_fn: impl FnOnce(&mut Vec<Strip>, &mut Vec<u8>),
) {
    match strip_storage.generation_mode {
        GenerationMode::Replace => strip_storage.strips.clear(),
        GenerationMode::Append => {}
        GenerationMode::ReplaceAfter(n) => strip_storage.strips.truncate(n),
    }

    if let Some(clip_path) = clip_path {
        temp_storage.clear();

        render_fn(&mut temp_storage.strips, &mut temp_storage.alphas);

        if clip_path.opaque_bbox.is_some_and(|opaque| {
            strip_bbox(&temp_storage.strips).is_some_and(|draw| {
                opaque.x0 <= draw.x0
                    && opaque.y0 <= draw.y0
                    && opaque.x1 >= draw.x1
                    && opaque.y1 >= draw.y1
            })
        }) {
            #[cfg(test)]
            CLIP_BYPASSES.with(|count| count.set(count.get() + 1));
            let alpha_offset = Strip::alpha_index(strip_storage.alphas.len());
            strip_storage
                .strips
                .extend(temp_storage.strips.iter().map(|strip| {
                    let mut strip = *strip;
                    strip.set_alpha_idx(strip.alpha_idx() + alpha_offset);
                    strip
                }));
            strip_storage.alphas.extend_from_slice(&temp_storage.alphas);
            return;
        }

        let path_data = PathDataRef {
            strips: &temp_storage.strips,
            alphas: &temp_storage.alphas,
            bbox: RectU32::new(0, 0, u32::MAX, u32::MAX),
            opaque_bbox: None,
        };
        intersect(level, clip_path, path_data, strip_storage);
    } else {
        render_fn(&mut strip_storage.strips, &mut strip_storage.alphas);
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use crate::fearless_simd::Level;
    use crate::kurbo::{Affine, Rect, Shape};
    use crate::peniko::Fill;
    use crate::strip_generator::{StripGenerator, StripStorage};

    #[test]
    fn reset() {
        let mut generator = StripGenerator::new(100, 100, Level::baseline());
        let mut storage = StripStorage::default();
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);

        generator.generate_filled_path(
            rect.to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );

        assert!(!generator.line_buf.is_empty());
        assert!(!storage.is_empty());

        generator.reset(100, 100);
        storage.clear();

        assert!(generator.line_buf.is_empty());
        assert!(storage.is_empty());
    }

    /// Assert that `generate_filled_rect_fast` produces the same strips as the
    /// path-based pipeline for the given rectangle.
    fn assert_rect_fast_eq_path(rect: Rect, test_name: &str) {
        let mut generator = StripGenerator::new(100, 100, Level::baseline());
        let mut storage_path = StripStorage::default();
        let mut storage_rect = StripStorage::default();

        generator.generate_filled_path(
            rect.to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage_path,
            None,
        );
        generator.reset(100, 100);

        generator.generate_filled_rect_fast(&rect, &mut storage_rect, None);

        assert_eq!(
            storage_path.strips, storage_rect.strips,
            "{test_name}: strips mismatch",
        );
        assert_eq!(
            storage_path.alphas, storage_rect.alphas,
            "{test_name}: alphas mismatch",
        );
    }

    #[test]
    fn rect_small_single_tile() {
        assert_rect_fast_eq_path(Rect::new(1.0, 1.0, 3.0, 3.0), "small_single_tile");
    }

    #[test]
    fn rect_spanning_multiple_tiles_horizontally() {
        assert_rect_fast_eq_path(Rect::new(2.0, 1.0, 14.0, 3.0), "spanning_horizontal");
    }

    #[test]
    fn rect_spanning_multiple_tiles_vertically() {
        assert_rect_fast_eq_path(Rect::new(1.0, 2.0, 3.0, 14.0), "spanning_vertical");
    }

    #[test]
    fn rect_spanning_multiple_tiles_both_directions() {
        assert_rect_fast_eq_path(Rect::new(2.0, 2.0, 18.0, 18.0), "spanning_both");
    }

    #[test]
    fn rect_tile_aligned() {
        assert_rect_fast_eq_path(Rect::new(0.0, 0.0, 8.0, 8.0), "tile_aligned");
    }

    #[test]
    fn rect_one_pixel_wide() {
        assert_rect_fast_eq_path(Rect::new(5.0, 2.0, 6.0, 12.0), "one_pixel_wide");
    }

    #[test]
    fn rect_one_pixel_tall() {
        assert_rect_fast_eq_path(Rect::new(2.0, 5.0, 12.0, 6.0), "one_pixel_tall");
    }

    #[test]
    fn rect_fractional_within_single_tile() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.25, 0.75, 2.5, 3.5),
            (1.2, 1.3, 1.8, 1.7),
            (0.1, 0.1, 3.9, 3.9),
            (2.5, 2.5, 2.6, 2.6),
            (0.01, 0.99, 3.99, 3.01),
        ];
        for (i, &(x0, y0, x1, y1)) in cases.iter().enumerate() {
            assert_rect_fast_eq_path(Rect::new(x0, y0, x1, y1), &format!("single_tile_{i}"));
        }
    }

    #[test]
    fn rect_fractional_multi_tile() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (1.5, 2.3, 10.7, 8.9),
            (0.5, 0.5, 8.5, 8.5),
            (2.3, 5.1, 15.7, 5.9),
            (5.1, 2.3, 5.9, 15.7),
            (0.25, 0.25, 12.75, 12.75),
            (1.0 / 3.0, 2.0 / 3.0, 10.33, 8.67),
            (1.99, 2.01, 9.01, 7.99),
            (3.9, 3.9, 8.1, 8.1),
            (3.2, 6.3, 14.8, 6.7),
            (6.3, 3.2, 6.7, 14.8),
            (0.1, 0.9, 49.9, 49.1),
            (4.0, 2.7, 12.0, 9.3),
            (2.7, 4.0, 9.3, 12.0),
            (1.5, 1.2, 10.5, 2.8),
            (1.5, 2.5, 14.5, 18.5),
            (0.7, 0.3, 30.2, 25.8),
            (7.9, 7.9, 8.1, 8.1),
            (3.5, 0.5, 4.5, 0.9),
            (0.01, 0.01, 99.99, 99.99),
            (10.0, 10.0, 10.1, 10.1),
        ];
        for (i, &(x0, y0, x1, y1)) in cases.iter().enumerate() {
            assert_rect_fast_eq_path(Rect::new(x0, y0, x1, y1), &format!("multi_tile_{i}"));
        }
    }

    #[test]
    fn rect_fractional_exhaustive() {
        for xi in 0..100_u32 {
            for yi in 0..100_u32 {
                let dx = xi as f64 * 0.01;
                let dy = yi as f64 * 0.01;
                let rect = Rect::new(dx, dy, 50.0 + dx, 50.0 + dy);
                assert_rect_fast_eq_path(rect, &format!("exhaustive_{dx}_{dy}"));
            }
        }
    }

    #[test]
    fn rect_inverted_both_axes() {
        assert_rect_fast_eq_path(Rect::new(18.0, 18.0, 2.0, 2.0), "inverted_both_axes");
    }
}

#[cfg(test)]
mod wide_tests {
    use super::{StripGenerator, StripStorage};
    use crate::{
        fearless_simd::Level,
        kurbo::{Affine, Rect, Shape},
        peniko::Fill,
    };

    fn sample(storage: &StripStorage, x: u32, y: u32) -> u8 {
        let row = y / 4 * 4;
        let start = storage.strips.partition_point(|strip| strip.y < row);
        for pair in storage.strips[start..].windows(2) {
            let (strip, next) = (pair[0], pair[1]);
            if strip.y != row || strip.is_sentinel() {
                break;
            }
            let end = strip.x + strip.width_to(&next);
            if strip.x <= x && x < end {
                return storage.alphas
                    [strip.alpha_idx() as usize + (x - strip.x) as usize * 4 + (y - row) as usize];
            }
            if end <= x && x < next.x && next.y == row && next.fill_gap() {
                return 255;
            }
        }
        0
    }

    #[test]
    fn sparse_source_tiles_allocate_for_geometry_instead_of_viewport_height() {
        let end = u32::MAX - 3;
        let mut generator = StripGenerator::new(end, end, Level::baseline());
        let bounded = |generator: &StripGenerator| {
            assert!(generator.tiles.windings.partial.capacity() <= 8);
            assert!(generator.tiles.windings.coarse.capacity() <= 8);
            assert!(generator.tiles.windings.active.capacity() <= 4);
        };
        bounded(&generator);
        for origin in [0.0, f64::from(end - 4)] {
            generator.reset(end, end);
            bounded(&generator);
            let rect = Rect::new(origin, origin, origin + 4.0, origin + 4.0);
            let mut storage = StripStorage::default();
            generator.generate_filled_rect_fast(&rect, &mut storage, None);
            assert!(!storage.strips.is_empty());
            bounded(&generator);
            generator.generate_filled_path(
                rect.to_path(0.1),
                Fill::NonZero,
                Affine::IDENTITY,
                None,
                &mut storage,
                None,
            );
            assert!(!storage.strips.is_empty());
            bounded(&generator);
            generator.generate_stroked_path(
                rect.to_path(0.1),
                &crate::kurbo::Stroke::new(1.0),
                Affine::IDENTITY,
                None,
                &mut storage,
                None,
            );
            assert!(!storage.strips.is_empty());
            bounded(&generator);
        }
        let mut storage = StripStorage::default();
        generator.generate_filled_path(
            core::iter::empty(),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );
        assert!(storage.strips.is_empty());
        bounded(&generator);
        let mut invalid = crate::kurbo::BezPath::new();
        invalid.move_to((0.0, 0.0));
        invalid.line_to((f64::NAN, 2.0));
        invalid.line_to((2.0, 2.0));
        invalid.close_path();
        generator.generate_filled_path(
            invalid,
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );
        assert!(storage.strips.is_empty());
        bounded(&generator);
    }

    #[test]
    fn tiny_far_clip_visits_only_its_source_window_and_preserves_grid_phase() {
        use crate::geometry::RectU32;
        let end = u32::MAX - 3;
        let grid = super::source_window_grid(
            RectU32::new(0, 0, end, end),
            RectU32::new(end - 4, end - 4, end, end),
        );
        assert_eq!(grid, RectU32::new(4294967280, 4294967280, end, end));
        assert_eq!(
            grid.width().div_ceil(super::SOURCE_WINDOW)
                * grid.height().div_ceil(super::SOURCE_WINDOW),
            1
        );
        assert_eq!(
            super::source_window_grid(
                RectU32::new(4, 8, 140000, 140000),
                RectU32::new(65538, 65542, 65540, 65544)
            ),
            RectU32::new(65536, 65540, 65540, 65544)
        );
        assert!(
            super::source_window_grid(RectU32::new(0, 0, 4, 4), RectU32::new(8, 8, 12, 12))
                .is_empty()
        );
    }

    #[test]
    fn fractional_edges_across_source_windows_keep_local_precision_on_both_axes() {
        for fraction in [0.002, 0.003, 1.0 / 256.0] {
            for vertical in [false, true] {
                let edge = super::SOURCE_WINDOW + 4;
                let end = f64::from(edge) + fraction;
                let rect = if vertical {
                    Rect::new(0.0, 0.0, 4.0, end)
                } else {
                    Rect::new(0.0, 0.0, end, 4.0)
                };
                let (width, height) = if vertical {
                    (4, edge + 4)
                } else {
                    (edge + 4, 4)
                };
                let mut generator = StripGenerator::new(width, height, Level::baseline());
                let mut storage = StripStorage::default();
                generator.generate_filled_path(
                    rect.to_path(0.1),
                    Fill::NonZero,
                    Affine::IDENTITY,
                    None,
                    &mut storage,
                    None,
                );
                let at = if vertical { (0, edge) } else { (edge, 0) };
                assert_eq!(
                    sample(&storage, at.0, at.1),
                    (fraction * 255.0 + 0.5) as u8,
                    "fraction={fraction} vertical={vertical}"
                );
                for coordinate in super::SOURCE_WINDOW - 2..super::SOURCE_WINDOW + 2 {
                    let at = if vertical {
                        (0, coordinate)
                    } else {
                        (coordinate, 0)
                    };
                    assert_eq!(sample(&storage, at.0, at.1), 255);
                }
            }
        }
    }

    #[test]
    fn two_dimensional_window_merge_preserves_holes_and_left_winding() {
        for rule in [Fill::NonZero, Fill::EvenOdd] {
            let mut path = Rect::new(-16.0, -16.0, 70000.0, 70000.0).to_path(0.1);
            let (lo, hi) = (65528.0, 65540.0);
            if rule == Fill::NonZero {
                path.move_to((lo, lo));
                path.line_to((lo, hi));
                path.line_to((hi, hi));
                path.line_to((hi, lo));
                path.close_path();
            } else {
                path.extend(Rect::new(lo, lo, hi, hi).path_elements(0.1));
            }
            let mut generator = StripGenerator::new(70004, 70004, Level::baseline());
            let mut storage = StripStorage::default();
            generator.generate_filled_path(path, rule, Affine::IDENTITY, None, &mut storage, None);
            assert_eq!(
                generator.source_window_replays, 4,
                "a two-by-two source grid replays the geometry exactly four times"
            );
            for y in 65524..65544 {
                for x in 65524..65544 {
                    let expected = if (65528..65540).contains(&x) && (65528..65540).contains(&y) {
                        0
                    } else {
                        255
                    };
                    assert_eq!(sample(&storage, x, y), expected, "{rule:?} ({x},{y})");
                }
            }
            assert_eq!(sample(&storage, 0, 0), 255);
            assert_eq!(sample(&storage, 69999, 69999), 255);
            assert_eq!(sample(&storage, 70000, 69999), 0);
        }
    }

    #[test]
    fn nonrectangular_clip_is_applied_after_global_window_merge() {
        let mut triangle = crate::kurbo::BezPath::new();
        triangle.move_to((65528.25, 65528.5));
        triangle.line_to((65540.75, 65529.25));
        triangle.line_to((65530.5, 65540.75));
        triangle.close_path();
        let mut generator = StripGenerator::new(70004, 70004, Level::baseline());
        let mut clip = StripStorage::default();
        generator.generate_filled_path(
            triangle,
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut clip,
            None,
        );
        let clip_ref = crate::clip::PathDataRef {
            strips: &clip.strips,
            alphas: &clip.alphas,
            bbox: crate::util::strip_bbox(&clip.strips).unwrap(),
            opaque_bbox: None,
        };
        let mut storage = StripStorage::default();
        generator.generate_filled_path(
            Rect::new(-16.0, -16.0, 70000.0, 70000.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            Some(clip_ref),
        );
        for y in 65524..65544 {
            for x in 65524..65544 {
                assert_eq!(sample(&storage, x, y), sample(&clip, x, y), "({x},{y})");
            }
        }
    }

    #[test]
    fn stroked_path_has_no_seam_between_source_windows() {
        let mut path = crate::kurbo::BezPath::new();
        path.move_to((-16.0, 3.25));
        path.line_to((70000.0, 3.25));
        let mut generator = StripGenerator::new(70004, 8, Level::baseline());
        let mut storage = StripStorage::default();
        generator.generate_stroked_path(
            path,
            &crate::kurbo::Stroke::new(1.0),
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );
        for x in super::SOURCE_WINDOW - 2..super::SOURCE_WINDOW + 2 {
            assert_eq!(sample(&storage, x, 2), 64, "x={x}");
            assert_eq!(sample(&storage, x, 3), 191, "x={x}");
            assert_eq!(sample(&storage, x, 4), 0, "x={x}");
        }
    }

    #[test]
    fn wide_extent_path_keeps_far_fractional_edge() {
        let mut generator = StripGenerator::new(70004, 4, Level::baseline());
        let mut storage = StripStorage::default();
        generator.generate_filled_path(
            Rect::new(0.0, 0.0, 70000.0 + 1.0 / 256.0, 4.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );
        let mut coverage = 0;
        for pair in storage.strips.windows(2) {
            let strip = pair[0];
            let width = strip.width_to(&pair[1]);
            if strip.x <= 70000 && 70000 < strip.x + width {
                coverage =
                    storage.alphas[strip.alpha_idx() as usize + (70000 - strip.x) as usize * 4];
            }
        }
        assert_eq!(coverage, 1);
    }

    #[test]
    fn wide_source_strips_keep_exact_translated_coverage() {
        for fast_rect in [false, true] {
            for (dx, dy) in [(70000, 0), (0, 70000), (70000, 70000)] {
                let mut small = StripGenerator::new(40, 40, Level::baseline());
                let mut wide = StripGenerator::new(dx + 40, dy + 40, Level::baseline());
                let mut reference = StripStorage::default();
                let mut translated = StripStorage::default();
                let rect = Rect::new(4.25, 4.5, 24.75, 12.25);
                let shifted = rect + crate::kurbo::Vec2::new(f64::from(dx), f64::from(dy));
                if fast_rect {
                    small.generate_filled_rect_fast(&rect, &mut reference, None);
                    wide.generate_filled_rect_fast(&shifted, &mut translated, None);
                } else {
                    small.generate_filled_path(
                        rect.to_path(0.1),
                        Fill::NonZero,
                        Affine::IDENTITY,
                        None,
                        &mut reference,
                        None,
                    );
                    wide.generate_filled_path(
                        shifted.to_path(0.1),
                        Fill::NonZero,
                        Affine::IDENTITY,
                        None,
                        &mut translated,
                        None,
                    );
                }
                assert!(!translated.strips.is_empty());
                for strip in &mut translated.strips {
                    if !strip.is_sentinel() {
                        strip.x -= dx;
                    }
                    strip.y -= dy;
                }
                assert_eq!(
                    translated.strips, reference.strips,
                    "fast_rect={fast_rect}, dx={dx}, dy={dy}"
                );
                assert_eq!(
                    translated.alphas, reference.alphas,
                    "fast_rect={fast_rect}, dx={dx}, dy={dy}"
                );
            }
        }
    }
}
