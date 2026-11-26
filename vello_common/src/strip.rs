// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.

use crate::flatten::Line;
use crate::geometry::RectU16;
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use crate::util::f32_to_u8;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use fearless_simd::*;

/// A strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strip {
    /// The x coordinate of the strip, in user coordinates.
    pub x: u16,
    /// The y coordinate of the strip, in user coordinates.
    pub y: u16,
    /// Packed alpha index and fill gap flag.
    ///
    /// Bit layout (u32):
    /// - bit 31: `fill_gap` (See `Strip::fill_gap()`).
    /// - bits 0..=30: `alpha_column` (See `Strip::alpha_column()`).
    packed_alpha_column_fill_gap: u32,
}

/// A fill region with alpha coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StripAlphaFillSegment {
    /// The fill region covered by this alpha segment.
    pub fill: StripFillSegment,
    /// The index of the first four-byte coverage column of the segment.
    pub alpha_column: u32,
}

impl Deref for StripAlphaFillSegment {
    type Target = StripFillSegment;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.fill
    }
}

impl DerefMut for StripAlphaFillSegment {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.fill
    }
}

/// A fill region without alpha coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StripFillSegment {
    /// The inclusive start x coordinate in tile units.
    pub tile_x0: u16,
    /// The exclusive end x coordinate in tile units.
    pub tile_x1: u16,
    /// The y coordinate in tile units.
    pub tile_y: u16,
}

impl StripFillSegment {
    /// The inclusive start x coordinate in pixels.
    #[inline(always)]
    pub const fn x0(self) -> u16 {
        self.tile_x0 * Tile::WIDTH
    }

    /// The exclusive end x coordinate in pixels.
    #[inline(always)]
    pub const fn x1(self) -> u32 {
        self.tile_x1 as u32 * Tile::WIDTH as u32
    }

    /// The y coordinate in pixels.
    #[inline(always)]
    pub const fn y(self) -> u16 {
        self.tile_y * Tile::HEIGHT
    }

    /// Return this segment's rectangle in tile coordinates.
    #[inline(always)]
    pub const fn tile_rect(self) -> RectU16 {
        RectU16::new(
            self.tile_x0,
            self.tile_y,
            self.tile_x1,
            self.tile_y.saturating_add(1),
        )
    }

    /// Return this segment's rectangle in pixel coordinates.
    #[inline(always)]
    pub const fn pixel_rect(self) -> RectU16 {
        RectU16::from_tile_bounds(self.tile_rect())
    }

    /// Return this segment's pixel-space rectangle shifted by `shift`.
    #[inline(always)]
    pub fn shift(self, shift: (i32, i32)) -> RectU16 {
        let shifted = |tile: u16, step: u16, delta: i32| {
            (i64::from(tile) * i64::from(step) + i64::from(delta)).clamp(0, i64::from(u16::MAX))
                as u16
        };
        RectU16::new(
            shifted(self.tile_x0, Tile::WIDTH, shift.0),
            shifted(self.tile_y, Tile::HEIGHT, shift.1),
            shifted(self.tile_x1, Tile::WIDTH, shift.0),
            shifted(self.tile_y + 1, Tile::HEIGHT, shift.1),
        )
    }
}

/// Iterate over all fill and alpha-fill regions formed by the sequence of strips,
/// within the tile-unit bounds indicated by `tile_bounds`.
pub fn visit_strip_fill_segments<C>(
    strips: &[Strip],
    tile_bounds: RectU16,
    context: &mut C,
    mut alpha_fill: impl FnMut(&mut C, StripAlphaFillSegment),
    mut fill: impl FnMut(&mut C, StripFillSegment),
) {
    // Need at least two strips: 1 (or more) for the generated path, and the sentinel strip.
    if strips.len() < 2 || tile_bounds.is_empty() {
        return;
    }

    for pair in strips.windows(2) {
        let strip = pair[0];
        let tile_y = strip.strip_y();

        // Skip strips that are outside the viewport vertically.
        if tile_y < tile_bounds.y0 {
            continue;
        }
        if tile_y >= tile_bounds.y1 {
            break;
        }

        let next_strip = pair[1];
        let strip_width = strip.width_to(&next_strip);

        debug_assert_eq!(
            strip.x % Tile::WIDTH,
            0,
            "strip x must be tile-width aligned",
        );
        debug_assert_eq!(
            strip_width % u32::from(Tile::WIDTH),
            0,
            "strip width must be tile-width aligned",
        );

        let strip_tile_x0 = strip.x / Tile::WIDTH;
        let strip_tile_x1 = u32::from(strip_tile_x0) + strip_width / u32::from(Tile::WIDTH);
        // Clip strips that are outside the viewport horizontally.
        let tile_x0 = strip_tile_x0.max(tile_bounds.x0);
        let tile_x1 = strip_tile_x1.min(u32::from(tile_bounds.x1)) as u16;

        if tile_x0 < tile_x1 {
            alpha_fill(
                context,
                StripAlphaFillSegment {
                    fill: StripFillSegment {
                        tile_x0,
                        tile_x1,
                        tile_y,
                    },
                    // Make sure to recalculate the index in case we had to clip.
                    alpha_column: strip.alpha_column()
                        + u32::from(tile_x0 - strip_tile_x0) * u32::from(Tile::WIDTH),
                },
            );
        }

        if next_strip.fill_gap() && next_strip.y == strip.y {
            // Similar procedure to above.

            let tile_x0 = strip_tile_x1.max(u32::from(tile_bounds.x0));
            let tile_x1 = u32::from((next_strip.x / Tile::WIDTH).min(tile_bounds.x1));

            if tile_x0 < tile_x1 {
                fill(
                    context,
                    StripFillSegment {
                        tile_x0: tile_x0 as u16,
                        tile_x1: tile_x1 as u16,
                        tile_y,
                    },
                );
            }
        }
    }
}

impl Strip {
    /// The bit mask for `fill_gap` packed into `packed_alpha_column_fill_gap`.
    const FILL_GAP_MASK: u32 = 1 << 31;

    /// Convert an aligned coverage buffer length to its packed column position.
    #[inline]
    pub fn column_from_byte_offset(bytes: usize) -> u32 {
        let height = usize::from(Tile::HEIGHT);
        assert!(bytes.is_multiple_of(height), "unaligned alpha byte offset");
        let column = bytes / height;
        assert!(
            column < Self::FILL_GAP_MASK as usize,
            "alpha column too large"
        );
        column as u32
    }

    /// Create a strip starting at an aligned byte offset in the coverage buffer.
    pub fn new(x: u16, y: u16, alpha_byte_offset: usize, fill_gap: bool) -> Self {
        let alpha_column = Self::column_from_byte_offset(alpha_byte_offset);
        let fill_gap = u32::from(fill_gap) << 31;
        Self {
            x,
            y,
            packed_alpha_column_fill_gap: alpha_column | fill_gap,
        }
    }

    /// Creates a sentinel strip.
    pub fn sentinel(y: u16, alpha_byte_offset: usize) -> Self {
        Self::new(u16::MAX, y, alpha_byte_offset, false)
    }

    /// Return whether the strip is a sentinel strip.
    pub fn is_sentinel(&self) -> bool {
        self.x == u16::MAX
    }

    /// Return the y coordinate of the strip, in strip units.
    pub fn strip_y(&self) -> u16 {
        self.y / Tile::HEIGHT
    }

    /// Returns the horizontal pixel width of this strip.
    ///
    /// **IMPORTANT**: This assumes that the `next` is actually the next adjacent strip
    /// to `self`, otherwise this method will return a garbage value!
    pub fn width_to(&self, next: &Self) -> u32 {
        next.alpha_column() - self.alpha_column()
    }

    /// Index of the first coverage column in the alpha buffer.
    #[inline(always)]
    pub fn alpha_column(&self) -> u32 {
        self.packed_alpha_column_fill_gap & !Self::FILL_GAP_MASK
    }

    /// Byte offset at which to slice the coverage buffer.
    #[inline(always)]
    pub fn alpha_byte_offset(self) -> usize {
        self.alpha_column() as usize * usize::from(Tile::HEIGHT)
    }

    /// Shift the strip by the given number of coverage columns.
    pub(crate) fn rebase(&mut self, columns: u32) {
        let column = self
            .alpha_column()
            .checked_add(columns)
            .expect("alpha column overflow");
        assert!(column < Self::FILL_GAP_MASK, "alpha column too large");
        self.packed_alpha_column_fill_gap =
            column | (self.packed_alpha_column_fill_gap & Self::FILL_GAP_MASK);
    }

    /// Returns whether the gap that lies between this strip and the previous in the same row should be filled.
    #[inline(always)]
    pub fn fill_gap(&self) -> bool {
        (self.packed_alpha_column_fill_gap & Self::FILL_GAP_MASK) != 0
    }

    /// Sets whether the gap that lies between this strip and the previous in the same row should be filled.
    #[inline(always)]
    pub fn set_fill_gap(&mut self, fill: bool) {
        let fill = u32::from(fill) << 31;
        self.packed_alpha_column_fill_gap =
            (self.packed_alpha_column_fill_gap & !Self::FILL_GAP_MASK) | fill;
    }

    /// When early culling is active, geometry fully to the left of the viewport creates no tiles.
    /// However, if that geometry has a non-zero winding (e.g. a large shape surrounding the
    /// viewport), then we must output strips for those fills.
    ///
    /// We reconstruct this "background" fill using `row_windings` (the winding at x=0) to emit solid
    /// strips for:
    ///      1. All rows vertically above the first visible tile.
    ///      2. 'Captive' rows between two tile-containing rows.
    ///      3. All rows vertically below the last visible tile.
    #[inline(always)]
    fn emit_culled_background<F>(
        start: u16,
        end: u16,
        viewport_width: u16,
        strips: &mut Vec<Self>,
        alphas: &mut Vec<u8>,
        windings: &crate::tile::CulledWindings,
        mut should_fill: F,
    ) where
        F: FnMut(i32) -> bool,
    {
        windings.for_active_rows_in_range(start as usize, end as usize, |row| {
            if should_fill(windings.coarse[row]) {
                let y_pos = row as u16 * Tile::HEIGHT;
                strips.push(Self::new(0, y_pos, alphas.len(), false));
                // TODO: Would be nice to get rid of this, but the current clipping code only
                // allows zero-width strips as a row terminator, not in-between.
                alphas.extend([255_u8; Tile::HEIGHT as usize * Tile::WIDTH as usize]);
                Self::emit_row_end(viewport_width, y_pos, true, strips, alphas);
            }
        });
    }

    fn emit_row_end(
        viewport_width: u16,
        y: u16,
        fill_gap: bool,
        strips: &mut Vec<Self>,
        alphas: &mut Vec<u8>,
    ) {
        let end = u32::from(viewport_width).next_multiple_of(u32::from(Tile::WIDTH));
        if end <= u32::from(u16::MAX) {
            strips.push(Self::new(end as u16, y, alphas.len(), fill_gap));
        } else if fill_gap {
            // An aligned endpoint of 65536 cannot be a strip start. Keep the
            // last tile explicit instead of aliasing the sentinel.
            let last = strips.last().unwrap();
            let alpha_end = u32::from(last.x)
                + (Self::column_from_byte_offset(alphas.len()) - last.alpha_column());
            if alpha_end < end {
                let last_x = u16::MAX / Tile::WIDTH * Tile::WIDTH;
                strips.push(Self::new(last_x, y, alphas.len(), true));
                alphas.extend([255_u8; Tile::HEIGHT as usize * Tile::WIDTH as usize]);
            }
        }
    }
}

// Clip to one pixel column before integrating. This bounds the horizontal
// displacement by one even when the original slope exceeds the f32 range.
#[cold]
fn wide_coverage<S: Simd>(
    s: S,
    line: Line,
    tile: Tile,
    sign: f32,
    locations: &mut [f32x4<S>; Tile::WIDTH as usize],
) -> f32x4<S> {
    let (a, b) = if line.p0.x <= line.p1.x {
        (line.p0, line.p1)
    } else {
        (line.p1, line.p0)
    };
    let (ax, ay, bx, by) = (
        f64::from(a.x),
        f64::from(a.y),
        f64::from(b.x),
        f64::from(b.y),
    );
    let y_at = |x| {
        if x == ax {
            ay
        } else if x == bx {
            by
        } else {
            crate::tile::crossing(ay, ax, by, bx, x)
        }
    };
    let mut acc = f32x4::splat(s, 0.);
    for (col, location) in locations.iter_mut().enumerate() {
        let left = f64::from(tile.x * Tile::WIDTH) + col as f64;
        let right = left + 1.;
        let x0 = ax.max(left);
        let x1 = bx.min(right);
        let mut area = [0.; 4];
        let mut height = [0.; 4];
        if x0 < x1 || (ax == bx && (left..right).contains(&ax)) {
            let y0 = y_at(x0);
            let y1 = if ax == bx { by } else { y_at(x1) };
            for row in 0..4 {
                let top = f64::from(tile.y * Tile::HEIGHT) + row as f64;
                let lo = y0.min(y1).max(top);
                let hi = y0.max(y1).min(top + 1.);
                if hi > lo {
                    let t = ((lo + hi) * 0.5 - y0) / (y1 - y0);
                    height[row] = (hi - lo) as f32 * sign;
                    area[row] = ((right - x0) - t * (x1 - x0)) as f32 * height[row];
                }
            }
        }
        *location += acc + f32x4::simd_from(s, area);
        acc += f32x4::simd_from(s, height);
    }
    acc
}

/// Render the tiles stored in `tiles` into the strip and alpha buffer.
pub fn render(
    level: Level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    dispatch!(level, simd => render_impl(simd,
                                         tiles,
                                         strip_buf,
                                         alpha_buf,
                                         fill_rule,
                                         aliasing_threshold,
                                         lines));
}

#[inline(always)]
fn render_impl<S: Simd>(
    s: S,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    aliasing_threshold: Option<u8>,
    lines: &[Line],
) {
    let row_windings = &tiles.windings.coarse;
    let has_culled_tiles = tiles.has_culled_tiles();
    let viewport_width = tiles.width();
    let strip_start = strip_buf.len();
    let maybe_emit_sentinel_strip = |strip_buf: &mut Vec<Strip>, alpha_buf: &Vec<u8>| {
        // Emit the final sentinel strip, if we produced at least one strip.
        if let Some(last_y) = strip_buf[strip_start..].last().map(|s| s.y) {
            strip_buf.push(Strip::sentinel(last_y, alpha_buf.len()));
        }
    };

    // If no tiles were culled and the tile buffer is empty, we can simply exit. If tiles were
    // culled, the tile buffer may be empty but there may be winding produced by culled geometry
    // left of the viewport that must be checked for filling.
    if !has_culled_tiles && tiles.is_empty() {
        return;
    }

    let should_fill = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    // Helper to handle "captive strips". When a row has tiles, but the first tile
    // is not at the left edge of the viewport (x != 0), we must emit a solid strip
    // from x=0 to that tile if the coarse winding dictates a fill.
    let emit_captive_strip =
        |y: u16, is_left_viewport: bool, strips: &mut Vec<Strip>, alphas: &mut Vec<u8>| {
            let coarse_wd = tiles.windings.coarse[y as usize];

            if should_fill(coarse_wd) && !is_left_viewport {
                strips.push(Strip::new(0, y * Tile::HEIGHT, alphas.len(), false));
                alphas.extend([255_u8; Tile::HEIGHT as usize * Tile::WIDTH as usize]);
            }

            let mut acc = f32x4::splat(s, coarse_wd as f32);
            if is_left_viewport {
                let fine_winding: f32x4<_> = tiles.windings.partial[y as usize].simd_into(s);
                acc += fine_winding;
            }

            (coarse_wd, acc)
        };

    // The accumulated tile winding delta. A line that crosses the top edge of a tile
    // increments the delta if the line is directed upwards, and decrements it if goes
    // downwards. Horizontal lines leave it unchanged.
    let mut winding_delta: i32 = 0;

    // The previous tile visited.
    let mut prev_tile = if has_culled_tiles && tiles.is_empty() {
        Tile::SENTINEL
    } else {
        *tiles.get(0)
    };

    // The accumulated (fractional) winding of the tile-sized location we're currently at.
    // Note multiple tiles can be at the same location.
    // Note that we are also implicitly assuming here that the tile height exactly fits into a
    // SIMD vector (i.e. 128 bits).
    let mut location_winding = [f32x4::splat(s, 0.0); Tile::WIDTH as usize];
    // The accumulated (fractional) windings at this location's right edge. When we move to the
    // next location, this is splatted to that location's starting winding.
    let mut accumulated_winding = f32x4::splat(s, 0.0);

    let left_viewport = prev_tile.x == 0;
    if has_culled_tiles {
        let row_max = prev_tile.y.min(row_windings.len() as u16);
        Strip::emit_culled_background(
            0,
            row_max,
            viewport_width,
            strip_buf,
            alpha_buf,
            &tiles.windings,
            should_fill,
        );
        if tiles.is_empty() {
            maybe_emit_sentinel_strip(strip_buf, alpha_buf);

            return;
        }
        let (wd, acc) = emit_captive_strip(prev_tile.y, left_viewport, strip_buf, alpha_buf);
        winding_delta = wd;
        accumulated_winding = acc;
        location_winding = [accumulated_winding; Tile::WIDTH as usize];
    }

    // The strip we're building.
    let mut strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        alpha_buf.len(),
        should_fill(winding_delta) && !left_viewport,
    );

    for (tile_idx, tile) in tiles.iter().copied().chain([Tile::SENTINEL]).enumerate() {
        let line = lines[tile.line_idx() as usize];
        let tile_left_x = f32::from(tile.x) * f32::from(Tile::WIDTH);
        let tile_top_y = f32::from(tile.y) * f32::from(Tile::HEIGHT);
        let p0_x = line.p0.x - tile_left_x;
        let p0_y = line.p0.y - tile_top_y;
        let p1_x = line.p1.x - tile_left_x;
        let p1_y = line.p1.y - tile_top_y;

        // Push out the winding as an alpha mask when we move to the next location (i.e., a tile
        // without the same location).
        if !prev_tile.same_loc(&tile) {
            match fill_rule {
                Fill::NonZero => {
                    let p1 = f32x4::splat(s, 0.5);
                    let p2 = f32x4::splat(s, 255.0);

                    #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                    for x in 0..Tile::WIDTH as usize {
                        let area = location_winding[x];
                        let coverage = area.abs();
                        let mulled = coverage.mul_add(p2, p1);
                        // Note that we are not storing the location winding here but the actual
                        // alpha value as f32, so we reuse the variable as a temporary storage.
                        // Also note that we need the `min` here because the winding can be > 1
                        // and thus the calculated alpha value need to be clamped to 255.
                        location_winding[x] = mulled.min(p2);
                    }
                }
                Fill::EvenOdd => {
                    let p1 = f32x4::splat(s, 0.5);
                    let p2 = f32x4::splat(s, -2.0);
                    let p3 = f32x4::splat(s, 255.0);

                    #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                    for x in 0..Tile::WIDTH as usize {
                        let area = location_winding[x];
                        let im1 = area.mul_add(p1, p1).floor();
                        let coverage = p2.mul_add(im1, area).abs();
                        let mulled = p3.mul_add(coverage, p1);
                        // TODO: It is possible that, unlike for `NonZero`, we don't need the `min`
                        // here.
                        location_winding[x] = mulled.min(p3);
                    }
                }
            };

            let p1 = s.combine_f32x4(location_winding[0], location_winding[1]);
            let p2 = s.combine_f32x4(location_winding[2], location_winding[3]);

            let mut u8_vals = f32_to_u8(s.combine_f32x8(p1, p2));

            if let Some(aliasing_threshold) = aliasing_threshold {
                u8_vals = s.select_u8x16(
                    u8_vals.simd_ge(u8x16::splat(s, aliasing_threshold)),
                    u8x16::splat(s, 255),
                    u8x16::splat(s, 0),
                );
            }

            alpha_buf.extend_from_slice(u8_vals.as_slice());

            #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
            for x in 0..Tile::WIDTH as usize {
                location_winding[x] = accumulated_winding;
            }
        }

        // Push out the strip if we're moving to a next strip.
        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            debug_assert_eq!(
                (prev_tile.x as u32 + 1) * Tile::WIDTH as u32 - strip.x as u32,
                Strip::column_from_byte_offset(alpha_buf.len()) - strip.alpha_column(),
                "The number of columns written to the alpha buffer should equal the number of columns spanned by this strip."
            );
            strip_buf.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            let left_viewport = tile.x == 0;
            if !prev_tile.same_row(&tile) {
                // Emit a final strip in the row if there is non-zero winding for the sparse fill
                if winding_delta != 0 {
                    Strip::emit_row_end(
                        viewport_width,
                        prev_tile.y * Tile::HEIGHT,
                        should_fill(winding_delta),
                        strip_buf,
                        alpha_buf,
                    );
                }

                // Logic identical to the start (see above): fill any vertical gaps (empty rows)
                // between the previous and current tile using the row windings.
                if has_culled_tiles && !is_sentinel {
                    Strip::emit_culled_background(
                        prev_tile.y + 1,
                        tile.y,
                        viewport_width,
                        strip_buf,
                        alpha_buf,
                        &tiles.windings,
                        should_fill,
                    );

                    let (wd, acc) = emit_captive_strip(tile.y, left_viewport, strip_buf, alpha_buf);
                    winding_delta = wd;
                    accumulated_winding = acc;
                } else {
                    winding_delta = 0;
                    accumulated_winding = f32x4::splat(s, 0.0);
                };

                #[expect(clippy::needless_range_loop, reason = "dimension clarity")]
                for x in 0..Tile::WIDTH as usize {
                    location_winding[x] = accumulated_winding;
                }
            } else {
                // Note: this fill is mathematically not necessary. It provides a way to reduce
                // accumulation of float rounding errors.
                accumulated_winding = f32x4::splat(s, winding_delta as f32);
            }

            if is_sentinel {
                break;
            }

            strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                alpha_buf.len(),
                should_fill(winding_delta) && !left_viewport,
            );
        }
        prev_tile = tile;

        // TODO: horizontal geometry has no impact on winding. This branch will be removed when
        // horizontal geometry is culled at the tile-generation stage.
        if line.p0.y == line.p1.y {
            continue;
        }

        // Lines moving upwards (in a y-down coordinate system) add to winding; lines moving
        // downwards subtract from winding.
        let sign = if line.p0.y > line.p1.y { 1. } else { -1. };

        // Calculate winding / pixel area coverage.
        //
        // Conceptually, horizontal rays are shot from left to right. Every time the ray crosses a
        // line that is directed upwards (decreasing `y`), the winding is incremented. Every time
        // the ray crosses a line moving downwards (increasing `y`), the winding is decremented.
        // The fractional area coverage of a pixel is the integral of the winding within it.
        //
        // Practically, to calculate this, each pixel is considered individually, and we determine
        // whether the line moves through this pixel. The line's y-delta within this pixel is
        // accumulated and added to the area coverage of pixels to the right. Within the pixel
        // itself, the area to the right of the line segment forms a trapezoid (or a triangle in
        // the degenerate case). The area of this trapezoid is added to the pixel's area coverage.
        //
        // For example, consider the following pixel square, with a line indicated by asterisks
        // starting inside the pixel and crossing its bottom edge. The area covered is the
        // trapezoid on the bottom-right enclosed by the line and the pixel square. The area is
        // positive if the line moves down, and negative otherwise.
        //
        //  __________________
        //  |                |
        //  |         *------|
        //  |        *       |
        //  |       *        |
        //  |      *         |
        //  |     *          |
        //  |    *           |
        //  |___*____________|
        //     *
        //    *

        let (line_top_y, line_top_x, line_bottom_y, line_bottom_x) = if p0_y < p1_y {
            (p0_y, p0_x, p1_y, p1_x)
        } else {
            (p1_y, p1_x, p0_y, p0_x)
        };

        let y_slope = (line_bottom_y - line_top_y) / (line_bottom_x - line_top_x);
        let x_slope = 1. / y_slope;

        winding_delta += sign as i32 * i32::from(tile.winding());

        // Reserve eleven fractional bits for pixel intersections. A very
        // shallow slope can also overflow the reciprocal or its y intercept.
        const LOCAL_LIMIT: f32 = 4096.;
        if !(p0_x.abs().max(p1_x.abs()).max(p0_y.abs()).max(p1_y.abs()) <= LOCAL_LIMIT
            && x_slope.abs() <= f32::MAX / (4. * LOCAL_LIMIT)
            && (y_slope.is_finite() || line.p0.x == line.p1.x))
        {
            accumulated_winding += wide_coverage(s, line, tile, sign, &mut location_winding);
            continue;
        }

        let line_top_y = f32x4::splat(s, line_top_y);
        let line_bottom_y = f32x4::splat(s, line_bottom_y);

        // See the explanation of this term on the `line_px_left_yx` and `line_px_right_yx`
        // variables below.
        let line_px_base_yx = line_top_y.mul_add(-x_slope, line_top_x);

        let px_top_y = f32x4::simd_from(s, [0., 1., 2., 3.]);
        let px_bottom_y = 1. + px_top_y;

        let ymin = line_top_y.max(px_top_y);
        let ymax = line_bottom_y.min(px_bottom_y);

        let mut acc = f32x4::splat(s, 0.0);

        for x_idx in 0..Tile::WIDTH {
            let x_idx_s = f32x4::splat(s, x_idx as f32);
            let px_left_x = x_idx_s;
            let px_right_x = 1.0 + x_idx_s;

            // The y-coordinate of the intersections between the line and the pixel's left and
            // right edges respectively.
            //
            // There is some subtlety going on here: `y_slope` will usually be finite, but will
            // be `inf` for purely vertical lines (`p0_x == p1_x`).
            //
            // In the case of `inf`, the resulting slope calculation will be `-inf` or `inf`
            // depending on whether the pixel edge is left or right of the line, respectively
            // (from the viewport's coordinate system perspective). The `min` and `max`
            // y-clamping logic generalizes nicely, as a pixel edge to the left of the line is
            // clamped to `ymin`, and a pixel edge to the right is clamped to `ymax`.
            //
            // In the special case where a vertical line and pixel edge are at the exact same
            // x-position (collinear), the line belongs to the pixel on whose _left_ edge it is
            // situated. The resulting slope calculation for the edge the line is situated on
            // will be NaN, as `0 * inf` results in NaN. This is true for both the left and
            // right edge.
            //
            // We know `ymin` and `ymax` are finite. We require the `max` operation to pick `ymin`
            // if its first operand is NaN. On x86, that maps to the semantics of `_mm_max_ps`,
            // which `f32x4::max` emits: that instruction takes element-wise
            // `if first > second { first } else { second }`. For AArch64, we do require the
            // `f32x4::max_precise` semantics (as `vmax_f32` returns NaN if either operand is NaN);
            // however, for AArch64 the precise version should be comparatively less expensive than
            // on x86. For `min`, we then know both operands are finite, so we can unambiguously
            // use the relaxed version. If this ever breaks, tests should fail loudly, because NaNs
            // happen a lot here!
            trait F32x4MaxExt {
                fn max_if_first_nan_take_second(self, rhs: Self) -> Self;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            impl<S: Simd> F32x4MaxExt for f32x4<S> {
                #[inline(always)]
                fn max_if_first_nan_take_second(self, rhs: Self) -> Self {
                    self.max(rhs)
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            impl<S: Simd> F32x4MaxExt for f32x4<S> {
                #[inline(always)]
                fn max_if_first_nan_take_second(self, rhs: Self) -> Self {
                    self.max_precise(rhs)
                }
            }
            let line_px_left_y = (px_left_x - line_top_x)
                .mul_add(y_slope, line_top_y)
                .max_if_first_nan_take_second(ymin)
                .min(ymax);
            let line_px_right_y = (px_right_x - line_top_x)
                .mul_add(y_slope, line_top_y)
                .max_if_first_nan_take_second(ymin)
                .min(ymax);

            // For each pixel we calculate the x-coordinates of the left- and rightmost points on
            // the line segment within that pixel. We do this based on the y-offsets of those two
            // points from the top of the line. This can be calculated as, e.g.,
            // `(line_px_left_y - line_top_y) * x_slope + line_top_x`.
            //
            // Rather than calculating that y-offset twice for each pixel within the loop through
            // subtracting from the points' y-coordinates, we get rid of that subtraction by baking
            // it into the algebraic "base" x-coordinate `line_px_base_yx` calculated above the
            // loop. When adding that term to `y * x_slope` it gives the x-coordinate of the point
            // along the line.
            //
            // Note `x_slope` is always finite, as horizontal geometry is elided.
            let line_px_left_yx = line_px_left_y.mul_add(x_slope, line_px_base_yx);
            let line_px_right_yx = line_px_right_y.mul_add(x_slope, line_px_base_yx);
            let h = (line_px_right_y - line_px_left_y).abs();

            // The trapezoidal area enclosed between the line and the right edge of the pixel
            // square. More straightforwardly written as follows, but the `madd` is faster.
            // 0.5 * h * (2. * px_right_x - line_px_right_yx - line_px_left_yx).
            let area = h * (line_px_right_yx + line_px_left_yx).mul_add(-0.5, px_right_x);
            location_winding[x_idx as usize] += area.mul_add(sign, acc);
            acc = h.mul_add(sign, acc);
        }

        accumulated_winding += acc;
    }

    if has_culled_tiles {
        Strip::emit_culled_background(
            (prev_tile.y + 1).min(row_windings.len() as u16),
            row_windings.len() as u16,
            viewport_width,
            strip_buf,
            alpha_buf,
            &tiles.windings,
            should_fill,
        );
    }

    maybe_emit_sentinel_strip(strip_buf, alpha_buf);
}

#[cfg(test)]
mod tests {
    use super::{Strip, StripFillSegment, visit_strip_fill_segments};
    use crate::geometry::RectU16;
    use alloc::vec;

    #[test]
    fn extreme_diagonal_half_coverage() {
        use super::*;
        use crate::flatten::Point;
        let far = f32::MAX;
        let a = Point { x: -far, y: -far };
        let b = Point { x: far, y: far };
        let c = Point { x: far, y: -far };
        let lines = [
            Line { p0: a, p1: b },
            Line { p0: b, p1: c },
            Line { p0: c, p1: a },
        ];
        let level = Level::baseline();
        let mut tiles = Tiles::new(level, 16, 16);
        tiles.make_tiles_analytic_aa(level, &lines, 16, 16);
        tiles.sort_tiles();
        let mut strips = vec![];
        let mut alphas = vec![];
        render(
            level,
            &tiles,
            &mut strips,
            &mut alphas,
            Fill::NonZero,
            None,
            &lines,
        );
        let strip = strips
            .windows(2)
            .find(|pair| {
                pair[0].y == 8
                    && pair[0].x <= 8
                    && u32::from(8 - pair[0].x) < pair[0].width_to(&pair[1])
            })
            .unwrap()[0];
        let i = strip.alpha_byte_offset() + usize::from(8 - strip.x) * 4;
        assert_eq!([alphas[i], alphas[i + 1], alphas[i + 4]], [128, 0, 255]);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn high_alpha_columns_round_trip_and_clip_without_a_backing_buffer() {
        assert_eq!(size_of::<Strip>(), 8);
        for boundary in [1_usize << 31, 1_usize << 32] {
            let mut strip = Strip::new(0, 0, boundary - 32, true);
            strip.rebase(4);
            let end = Strip::sentinel(0, boundary + 16);
            assert!(strip.fill_gap());
            assert!(!end.fill_gap());
            assert_eq!(strip.alpha_byte_offset(), boundary - 16);
            assert_eq!(end.alpha_byte_offset(), boundary + 16);
            assert_eq!(strip.width_to(&end), 8);
            let mut clipped = None;
            visit_strip_fill_segments(
                &[strip, end],
                RectU16::new(1, 0, 2, 1),
                &mut clipped,
                |out, segment| *out = Some(segment),
                |_, _| unreachable!(),
            );
            assert_eq!(clipped.unwrap().alpha_column as usize * 4, boundary);
        }
        let last = Strip::sentinel(0, (1_usize << 33) - 4);
        assert_eq!(last.alpha_column(), (1 << 31) - 1);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "alpha column too large")]
    fn eight_gibibyte_alpha_position_is_rejected() {
        Strip::sentinel(0, 1_usize << 33);
    }

    #[test]
    fn full_domain_alpha_strip_is_clipped_before_narrowing() {
        let strips = [Strip::new(0, 0, 0, false), Strip::sentinel(0, 65536 * 4)];
        assert_eq!(strips[0].width_to(&strips[1]), 65536);
        for width in 65532..=u16::MAX {
            let mut segments = vec![];
            visit_strip_fill_segments(
                &strips,
                RectU16::new(0, 0, width, 1).covering_tiles(),
                &mut segments,
                |segments, segment| segments.push(segment),
                |_, _| panic!("dense strip must not produce sparse fills"),
            );
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0].tile_x0, 0);
            assert_eq!(segments[0].tile_x1, width.div_ceil(4));
            assert_eq!(segments[0].alpha_column, 0);
        }
    }

    #[test]
    fn segment_shift_precedes_physical_domain_clipping() {
        let segment = StripFillSegment {
            tile_x0: 16383,
            tile_x1: 16384,
            tile_y: 16383,
        };
        assert_eq!(segment.x1(), 65536);
        assert_eq!(
            segment.shift((-4, -4)),
            RectU16::new(65528, 65528, 65532, 65532)
        );
    }

    #[test]
    fn full_domain_row_end_keeps_last_tile_without_sentinel_alias() {
        let mut strips = vec![Strip::new(0, 0, 0, false)];
        let mut alphas = vec![255; 16];
        Strip::emit_row_end(u16::MAX, 0, true, &mut strips, &mut alphas);
        assert_eq!(strips.len(), 2);
        assert_eq!(strips[1].x, 65532);
        assert!(!strips[1].is_sentinel());
        assert!(strips[1].fill_gap());
        assert_eq!(alphas.len(), 32);
        Strip::emit_row_end(u16::MAX, 0, true, &mut strips, &mut alphas);
        assert_eq!(
            strips.len(),
            2,
            "an already present last tile is not duplicated"
        );
    }
}
