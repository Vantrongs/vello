// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::clip::{ClipRef, ClipShape, PathDataRef, intersect};
use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::geometry::RectU16;
use crate::kurbo::{Affine, PathEl, Rect, Stroke};
use crate::peniko::Fill;
use crate::strip::Strip;
use crate::tile::Tiles;
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
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    temp_storage: StripStorage,
    tiles: Tiles,
    width: u16,
    height: u16,
    /// The part of the viewport whose strips are generated, if not all of it.
    window: Option<RectU16>,
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level, width, height),
            flatten_ctx: FlattenCtx::default(),
            stroke_ctx: StrokeCtx::default(),
            temp_storage: StripStorage::default(),
            width,
            height,
            window: None,
        }
    }

    /// Create a strip generator for a viewport of `width` by `height` that generates
    /// its strips in `window`, a part of it whose top left lies on a tile corner, at the
    /// window's origin, with the memory of the window: those the viewport has there,
    /// with their fills, see [`Tiles::make_window_tiles_analytic_aa`].
    ///
    /// Paths and rectangles are given, culled and rounded in the viewport's
    /// coordinates, and so are the bounds of clip paths, whose strips are those of the
    /// window. A dashed stroke draws only the dashes that can reach the window, see
    /// [`flatten::stroke_near`].
    pub fn new_window(width: u16, height: u16, window: RectU16, level: Level) -> Self {
        Self {
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level, window.width(), window.height()),
            flatten_ctx: FlattenCtx::default(),
            stroke_ctx: StrokeCtx::default(),
            temp_storage: StripStorage::default(),
            width,
            height,
            window: Some(window),
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
            .unwrap_or(RectU16::new(0, 0, self.width, self.height));
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
            cull_bbox,
        );

        self.generate_with_clip(aliasing_threshold, strip_storage, fill_rule, clip_path);
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
            .unwrap_or(RectU16::new(0, 0, self.width, self.height));
        match self.window {
            None => flatten::stroke(
                self.level,
                path,
                stroke,
                transform,
                &mut self.line_buf,
                &mut self.flatten_ctx,
                &mut self.stroke_ctx,
                cull_bbox,
            ),
            // What lies outside the window is closed outlines, whose winding cancels in it.
            Some(window) => flatten::stroke_near(
                self.level,
                path,
                stroke,
                transform,
                &mut self.line_buf,
                &mut self.flatten_ctx,
                &mut self.stroke_ctx,
                cull_bbox,
                window,
            ),
        }
        self.generate_with_clip(aliasing_threshold, strip_storage, Fill::NonZero, clip_path);
    }

    fn generate_with_clip(
        &mut self,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        fill_rule: Fill,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        match self.window {
            None => self.tiles.make_tiles_analytic_aa(
                self.level,
                &self.line_buf,
                self.width,
                self.height,
            ),
            Some(window) => self.tiles.make_window_tiles_analytic_aa(
                self.level,
                &self.line_buf,
                self.width,
                self.height,
                window,
            ),
        };

        self.tiles.sort_tiles();

        let level = self.level;
        let tiles = &self.tiles;
        let line_buf = &self.line_buf;
        let windowed = self.window.is_some();
        render_with_clip(
            level,
            &mut self.temp_storage,
            strip_storage,
            clip_path,
            |strips, alphas| {
                if windowed {
                    strip::render_window(
                        level,
                        tiles,
                        strips,
                        alphas,
                        fill_rule,
                        aliasing_threshold,
                        line_buf,
                    );
                } else {
                    strip::render(
                        level,
                        tiles,
                        strips,
                        alphas,
                        fill_rule,
                        aliasing_threshold,
                        line_buf,
                    );
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
        let window = self.window;
        render_with_clip(
            level,
            &mut self.temp_storage,
            strip_storage,
            complex_clip,
            |strips, alphas| match window {
                None => rect::render(level, clamped, strips, alphas),
                Some(window) => rect::render_window(level, clamped, window, strips, alphas),
            },
        );

        shape
    }

    /// Reset the strip generator for a viewport size, resizing only when needed.
    pub fn reset(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.line_buf.clear();
        self.tiles.reset(width, height);
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
    let alpha_offset = target.alphas.len() as u32;
    target.strips.extend(source.strips.iter().map(|strip| {
        let mut strip = *strip;
        strip.set_alpha_idx(strip.alpha_idx() + alpha_offset);
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
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::format;
    use alloc::vec::Vec;

    use crate::clip::{ClipRef, ClipShape, PathDataRef};
    use crate::fearless_simd::Level;
    use crate::geometry::RectU16;
    use crate::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
    use crate::peniko::Fill;
    use crate::strip::visit_strip_fill_segments;
    use crate::strip_generator::{StripGenerator, StripStorage};
    use crate::tile::Tile;
    use crate::util::strip_bbox;

    /// A source of pseudo-random numbers.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, end: u64) -> u64 {
            self.next() % end
        }

        fn number(&mut self, low: f64, high: f64) -> f64 {
            low + (self.next() >> 11) as f64 / (1_u64 << 53) as f64 * (high - low)
        }
    }

    /// Something drawn, or pushed as a clip.
    enum Drawn {
        Fill(BezPath, Fill),
        Stroke(BezPath, Stroke),
        Rect(Rect),
    }

    /// Return a coordinate around a viewport side of `size`: mostly near it, some on
    /// tile edges or a little to either side, which `f32` rounds onto them, some far
    /// beyond it, where fills are clipped before flattening.
    fn coordinate(random: &mut Random, size: u16) -> f64 {
        let size = f64::from(size);
        let edge = |random: &mut Random| (random.below(size as u64 / 4 + 2) * 4) as f64;
        match random.below(8) {
            0 => edge(random),
            1 => {
                let off = [
                    2_f64.powi(-22),
                    -(2_f64.powi(-22)),
                    1e-7,
                    -1e-7,
                    2_f64.powi(-30),
                ];
                edge(random) + off[random.below(off.len() as u64) as usize]
            }
            2 => random.number(-1e6, 1e6),
            _ => random.number(-40.0, size + 40.0),
        }
    }

    fn random_drawn(random: &mut Random, (width, height): (u16, u16)) -> Drawn {
        let point = |random: &mut Random| (coordinate(random, width), coordinate(random, height));
        if random.below(5) == 0 {
            let (a, mut b) = (point(random), point(random));
            // Sides whose distance rounds away in `f32`, which still cover a tile.
            match random.below(4) {
                0 => b.0 = a.0 + 1e-7,
                1 => b.1 = a.1 - 1e-7,
                _ => {}
            }
            return Drawn::Rect(Rect::new(a.0, a.1, b.0, b.1));
        }
        let mut path = BezPath::new();
        path.move_to(point(random));
        for _ in 0..1 + random.below(8) {
            match random.below(6) {
                0 => path.quad_to(point(random), point(random)),
                1 => path.curve_to(point(random), point(random), point(random)),
                2 => path.close_path(),
                _ => path.line_to(point(random)),
            }
        }
        match random.below(3) {
            0 => Drawn::Stroke(path, Stroke::new(random.number(0.5, 12.0))),
            1 => Drawn::Fill(path, Fill::EvenOdd),
            _ => Drawn::Fill(path, Fill::NonZero),
        }
    }

    /// Draw `drawn` with `generator` under `clip`, whose strips cull to `cull`, and
    /// return its strips and the shape it has as a clip.
    fn draw(
        generator: &mut StripGenerator,
        drawn: &Drawn,
        clip: Option<(&StripStorage, ClipShape)>,
        cull: RectU16,
    ) -> (StripStorage, ClipShape) {
        let clip = clip.map(|(storage, shape)| ClipRef {
            path: PathDataRef {
                strips: &storage.strips,
                alphas: &storage.alphas,
                bbox: cull,
                opaque_bbox: None,
            },
            shape,
        });
        let mut storage = StripStorage::default();
        let shape = match drawn {
            Drawn::Rect(rect) => generator.generate_filled_rect_fast(rect, &mut storage, clip),
            Drawn::Fill(path, fill_rule) => {
                generator.generate_filled_path(
                    path,
                    *fill_rule,
                    Affine::IDENTITY,
                    None,
                    &mut storage,
                    clip.map(|clip| clip.path),
                );
                ClipShape::Path
            }
            Drawn::Stroke(path, stroke) => {
                generator.generate_stroked_path(
                    path,
                    stroke,
                    Affine::IDENTITY,
                    None,
                    &mut storage,
                    clip.map(|clip| clip.path),
                );
                ClipShape::Path
            }
        };
        (storage, shape)
    }

    fn clip_of(clip: &Option<(StripStorage, ClipShape)>) -> Option<(&StripStorage, ClipShape)> {
        clip.as_ref().map(|(storage, shape)| (storage, *shape))
    }

    /// Return the alphas of each tile `strips` cover in `tiles`, those of a fill opaque,
    /// at their offset from the top left of `tiles`.
    fn covered(storage: &StripStorage, tiles: RectU16) -> BTreeMap<(u16, u16), [u8; 16]> {
        let mut covered = BTreeMap::new();
        let at = |x: u16, y: u16| (x - tiles.x0, y - tiles.y0);
        visit_strip_fill_segments(
            &storage.strips,
            tiles,
            &mut covered,
            |covered, segment| {
                for (column, x) in (segment.tile_x0..segment.tile_x1).enumerate() {
                    let start = segment.alpha_idx as usize + column * 16;
                    let alphas = storage.alphas[start..start + 16].try_into().unwrap();
                    covered.insert(at(x, segment.tile_y), alphas);
                }
            },
            |covered, segment| {
                for x in segment.tile_x0..segment.tile_x1 {
                    covered.insert(at(x, segment.tile_y), [255; 16]);
                }
            },
        );
        covered
    }

    /// A generator for a window of a viewport makes the strips and fills a generator for
    /// the whole viewport makes there, and their alphas in rows with no strips left of
    /// the window, for paths, strokes and rectangles reaching past the viewport's edges
    /// and far beyond, under clip paths and rectangles.
    #[test]
    fn a_window_holds_the_strips_of_the_viewport() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        // Windows compared that start right of the viewport's left edge, and at it.
        let mut compared = [0; 2];
        for case in 0..3000 {
            let size = (1 + random.below(160) as u16, 1 + random.below(160) as u16);
            let start = |random: &mut Random, size: u16| match random.below(3) {
                0 => 0,
                _ => random.below(u64::from(size.div_ceil(Tile::WIDTH))) as u16 * Tile::WIDTH,
            };
            let (x0, y0) = (start(&mut random, size.0), start(&mut random, size.1));
            let end = |random: &mut Random, start: u16, size: u16| match random.below(3) {
                0 => size,
                _ => start + 1 + random.below(u64::from(size - start)) as u16,
            };
            let window = RectU16::new(
                x0,
                y0,
                end(&mut random, x0, size.0),
                end(&mut random, y0, size.1),
            );
            let clips: Vec<Drawn> = (0..random.below(3))
                .map(|_| random_drawn(&mut random, size))
                .collect();
            let item = random_drawn(&mut random, size);
            let mut whole = StripGenerator::new(size.0, size.1, Level::baseline());
            let mut part = StripGenerator::new_window(size.0, size.1, window, Level::baseline());
            let (mut whole_clip, mut part_clip) = (None, None);
            let tiles = window.to_tile_bounds();
            // The rows of the window where what lies left of it changes alphas in it: of
            // paths with tiles left of it, or lines left of the viewport, whose fractional
            // winding the renderer adds only to a first tile in the first column.
            let mut left_rows = BTreeSet::new();
            let mut cull = RectU16::new(0, 0, size.0, size.1);
            for drawn in clips.iter().chain([&item]) {
                let drawn_whole = draw(&mut whole, drawn, clip_of(&whole_clip), cull);
                let drawn_part = draw(&mut part, drawn, clip_of(&part_clip), cull);
                if !matches!(drawn, Drawn::Rect(_)) {
                    let rows = tiles.y0..tiles.y1;
                    for tile in whole.tiles.iter() {
                        if tile.x < tiles.x0 && rows.contains(&tile.y) {
                            left_rows.insert(tile.y - tiles.y0);
                        }
                    }
                    if whole.tiles.has_culled_tiles() {
                        let windings = &whole.tiles.windings;
                        windings.for_active_rows_in_range(
                            usize::from(tiles.y0),
                            usize::from(tiles.y1),
                            |row| {
                                left_rows.insert(row as u16 - tiles.y0);
                            },
                        );
                    }
                }
                // The clip's bounds, as `ClipContext` keeps them.
                cull = strip_bbox(&drawn_whole.0.strips).unwrap_or(RectU16::ZERO);
                (whole_clip, part_clip) = (Some(drawn_whole), Some(drawn_part));
            }
            let expected = covered(&whole_clip.unwrap().0, tiles);
            let actual = covered(
                &part_clip.unwrap().0,
                RectU16::new(0, 0, tiles.width(), tiles.height()),
            );
            let keys = |covered: &BTreeMap<(u16, u16), [u8; 16]>| {
                covered.keys().copied().collect::<Vec<_>>()
            };
            assert_eq!(
                keys(&actual),
                keys(&expected),
                "case {case}: {window:?} of {size:?}"
            );
            let kept = |covered: &BTreeMap<(u16, u16), [u8; 16]>| {
                covered
                    .iter()
                    .filter(|((_, y), _)| !left_rows.contains(y))
                    .map(|(tile, alphas)| (*tile, *alphas))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                kept(&actual),
                kept(&expected),
                "case {case}: {window:?} of {size:?}"
            );
            if !expected.is_empty() {
                compared[usize::from(window.x0 == 0)] += 1;
            }
        }
        assert!(compared.iter().all(|count| *count > 300), "{compared:?}");
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
