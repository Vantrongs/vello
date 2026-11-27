// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::clip::{ClipRef, ClipShape, PathDataRef, intersect};
use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::geometry::RectU16;
use crate::kurbo::{Affine, PathEl, Rect, Stroke};
use crate::path_budget::{CapacityExceeded, Meter, PathBudget};
use crate::peniko::Fill;
use crate::strip::Strip;
use crate::stroke::StrokeBuffers;
use crate::tile::Tiles;
use crate::util::strip_bbox;
use crate::{flatten, rect, strip};
use alloc::vec::Vec;

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
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeBuffers,
    temp_storage: StripStorage,
    tiles: Tiles,
    width: u16,
    height: u16,
}

/// Geometry and winding rule for a prepared path.
#[derive(Clone, Copy, Debug)]
pub enum PathStyle<'a> {
    /// Fill with the specified winding rule.
    Fill(Fill),
    /// Expand a stroke before rasterization.
    Stroke(&'a Stroke),
}

/// Directed edges prepared for the clip and viewport used at preparation time.
#[derive(Debug)]
pub struct PreparedPath<'g, 'c> {
    generator: &'g mut StripGenerator,
    clip: Option<PathDataRef<'c>>,
    fill: Fill,
}

impl PreparedPath<'_, '_> {
    pub(crate) fn fill_rule(&self) -> Fill {
        self.fill
    }

    /// Device-space directed edges for shading or inspection.
    pub fn edges(&self) -> &[Line] {
        &self.generator.line_buf
    }

    /// Rasterize within the additional capacity allowance. Refusal preserves all
    /// previous strips and alpha bytes; scratch capacity remains reusable.
    pub fn try_submit(
        self,
        threshold: Option<u8>,
        storage: &mut StripStorage,
        budget: &PathBudget,
    ) -> Result<(), CapacityExceeded> {
        let meter = Meter::Raster(budget);
        let g = self.generator;
        if meter.failed() {
            return Err(CapacityExceeded);
        }
        g.tiles
            .make_tiles_metered(g.level, &g.line_buf, g.width, g.height, meter);
        if meter.failed() {
            return Err(CapacityExceeded);
        }
        g.tiles.sort_tiles();
        let render = |target: &mut StripStorage| {
            strip::render_metered(
                g.level,
                &g.tiles,
                &mut target.strips,
                &mut target.alphas,
                self.fill,
                threshold,
                &g.line_buf,
                meter,
            );
        };
        let start = storage.strips.len();
        let alpha_start = storage.alphas.len();
        if let Some(clip) = self.clip {
            g.temp_storage.clear();
            render(&mut g.temp_storage);
            if !meter.failed() && within_opaque_clip(&clip, &g.temp_storage.strips) {
                if meter.room(&mut storage.strips, g.temp_storage.strips.len())
                    && meter.room(&mut storage.alphas, g.temp_storage.alphas.len())
                {
                    append_rebased(storage, &g.temp_storage);
                }
            } else if !meter.failed() {
                crate::clip::intersect_metered(
                    g.level,
                    clip,
                    PathDataRef {
                        strips: &g.temp_storage.strips,
                        alphas: &g.temp_storage.alphas,
                        bbox: RectU16::new(0, 0, g.width, g.height),
                        opaque_bbox: None,
                    },
                    storage,
                    meter,
                );
            }
        } else {
            render(storage);
        }
        if meter.failed() {
            storage.strips.truncate(start);
            storage.alphas.truncate(alpha_start);
            return Err(CapacityExceeded);
        }
        let retain = match storage.generation_mode {
            GenerationMode::Append => start,
            GenerationMode::Replace => 0,
            GenerationMode::ReplaceAfter(n) => n.min(start),
        };
        storage.strips.drain(retain..start);
        Ok(())
    }

    /// Rasterize the inspected edges into strip storage.
    pub fn submit(self, aliasing_threshold: Option<u8>, storage: &mut StripStorage) {
        self.generator
            .generate_with_clip(aliasing_threshold, storage, self.fill, self.clip);
    }
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level, width, 0),
            flatten_ctx: FlattenCtx::default(),
            stroke_ctx: StrokeBuffers::default(),
            temp_storage: StripStorage::default(),
            width,
            height,
        }
    }

    /// Get this strip generator's viewport width.
    #[inline(always)]
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get this strip generator's viewport height.
    #[inline(always)]
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Pixel bounds used for path culling under the given accumulated clip.
    pub fn clip_bounds(&self, clip_path: Option<PathDataRef<'_>>) -> RectU16 {
        clip_path
            .map(|clip_path| clip_path.bbox)
            .unwrap_or(RectU16::new(0, 0, self.width, self.height))
    }

    /// Prepare directed edges once. The budget covers additional Vello-owned buffer
    /// capacity, including stroke expansion.
    pub fn prepare_path<'g, 'c>(
        &'g mut self,
        path: impl IntoIterator<Item = PathEl>,
        style: PathStyle<'_>,
        transform: Affine,
        clip: Option<PathDataRef<'c>>,
        budget: &PathBudget,
    ) -> Result<PreparedPath<'g, 'c>, CapacityExceeded> {
        let meter = Meter::Indexed(budget);
        if meter.failed() {
            return Err(CapacityExceeded);
        }
        let prepared = self.prepare(path, style, transform, clip, meter);
        if meter.failed() {
            Err(CapacityExceeded)
        } else {
            Ok(prepared)
        }
    }

    fn prepare<'g, 'c>(
        &'g mut self,
        path: impl IntoIterator<Item = PathEl>,
        style: PathStyle<'_>,
        transform: Affine,
        clip: Option<PathDataRef<'c>>,
        meter: Meter<'_>,
    ) -> PreparedPath<'g, 'c> {
        let bbox = self.clip_bounds(clip);
        let fill = match style {
            PathStyle::Fill(fill) => {
                fearless_simd::dispatch!(self.level, simd => flatten::fill_path::<_, true>(
                    simd, path, transform, &mut self.line_buf, &mut self.flatten_ctx, bbox, meter
                ));
                fill
            }
            PathStyle::Stroke(stroke) => {
                flatten::stroke_metered(
                    self.level,
                    path,
                    stroke,
                    transform,
                    &mut self.line_buf,
                    &mut self.flatten_ctx,
                    flatten::StrokeExpansion::Vello(&mut self.stroke_ctx),
                    bbox,
                    meter,
                );
                Fill::NonZero
            }
        };
        PreparedPath {
            generator: self,
            clip,
            fill,
        }
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
        self.prepare(
            path,
            PathStyle::Fill(fill_rule),
            transform,
            clip_path,
            Meter::default(),
        )
        .submit(aliasing_threshold, strip_storage);
    }

    /// Prepare and submit a stroke using the same edge coordinates as ordinary rasterization.
    /// Indexed edges for sample queries are clipped to the view before rounding to f32.
    pub fn try_generate_stroke(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        threshold: Option<u8>,
        storage: &mut StripStorage,
        clip: Option<PathDataRef<'_>>,
        budget: &PathBudget,
    ) -> Result<(), CapacityExceeded> {
        self.prepare(
            path,
            PathStyle::Stroke(stroke),
            transform,
            clip,
            Meter::Raster(budget),
        )
        .try_submit(threshold, storage, budget)
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
        self.prepare(
            path,
            PathStyle::Stroke(stroke),
            transform,
            clip_path,
            Meter::default(),
        )
        .submit(aliasing_threshold, strip_storage);
    }

    fn generate_with_clip(
        &mut self,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        fill_rule: Fill,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        self.tiles
            .make_tiles_analytic_aa(self.level, &self.line_buf, self.width, self.height);

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
                strip::render(
                    level,
                    tiles,
                    strips,
                    alphas,
                    fill_rule,
                    aliasing_threshold,
                    line_buf,
                );
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
    pub fn reset(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.line_buf.clear();
        self.tiles.reset(width, 0);
        self.temp_storage.clear();
    }
}

/// Whether `strips` lie entirely where `clip` is proven fully opaque, so intersecting
/// them with it would change nothing.
pub(crate) fn within_opaque_clip(clip: &PathDataRef<'_>, strips: &[Strip]) -> bool {
    clip.opaque_bbox.is_some_and(|opaque| {
        strip_bbox(strips).is_some_and(|draw| {
            opaque.x0 <= draw.x0
                && opaque.y0 <= draw.y0
                && opaque.x1 >= draw.x1
                && opaque.y1 >= draw.y1
        })
    })
}

/// Appends the strips of `source` to `target`, rebasing their alpha indices.
fn append_rebased(target: &mut StripStorage, source: &StripStorage) {
    // The room pushing the strips one at a time, as intersecting them with the clip
    // does, would reach, each growth doubling, rather than what extending reserves.
    let len = target.strips.len() + source.strips.len();
    while target.strips.capacity() < len {
        let full = target.strips.capacity() - target.strips.len();
        target.strips.reserve(full + 1);
    }
    Strip::column_from_byte_offset(
        target
            .alphas
            .len()
            .checked_add(source.alphas.len())
            .expect("alpha byte length overflow"),
    );
    let columns = Strip::column_from_byte_offset(target.alphas.len());
    target.strips.extend(source.strips.iter().map(|strip| {
        let mut strip = *strip;
        strip.rebase(columns);
        strip
    }));
    target.alphas.extend_from_slice(&source.alphas);
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

        if within_opaque_clip(&clip_path, &temp_storage.strips) {
            append_rebased(strip_storage, temp_storage);
            return;
        }

        let path_data = PathDataRef {
            strips: &temp_storage.strips,
            alphas: &temp_storage.alphas,
            bbox: RectU16::new(0, 0, u16::MAX, u16::MAX),
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
    use crate::strip::Strip;
    use crate::strip_generator::{StripGenerator, StripStorage, append_rebased};

    /// Strips appended inside an opaque clip take the room that pushing them one at a
    /// time, as intersecting them with the clip does, would take.
    #[test]
    fn appended_strips_take_the_room_pushed_ones_would() {
        let strip = |i: usize| Strip::new(i as u16, 0, 0, false);
        for (before, added) in [(0, 1), (3, 2), (6, 7), (5, 100), (1000, 3000)] {
            let mut target = StripStorage::default();
            for i in 0..before {
                target.strips.push(strip(i));
            }
            let mut source = StripStorage::default();
            source.strips.extend((0..added).map(strip));
            let mut pushed = target.strips.clone();
            pushed.reserve_exact(target.strips.capacity() - pushed.capacity());
            for i in 0..added {
                pushed.push(strip(i));
            }
            append_rebased(&mut target, &source);
            assert_eq!(target.strips.len(), before + added);
            assert_eq!(
                target.strips.capacity(),
                pushed.capacity(),
                "{before} strips, {added} added"
            );
        }
    }

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
