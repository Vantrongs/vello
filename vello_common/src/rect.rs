// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fast pixel-aligned rectangle rendering directly into strips.

use crate::kurbo::Rect;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::simd::element_wise_splat;
use crate::strip::Strip;
use crate::tile::Tile;
use crate::util::f32_to_u8;
use alloc::vec::Vec;
use fearless_simd::*;

/// Render a pixel-aligned rectangle directly into strips.
///
/// This bypasses the full path processing pipeline (flatten → tiles → strips)
/// by directly creating strip coverage data for the rectangle.
///
/// The rect bounds should already be clamped to the viewport.
pub fn render(level: Level, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
    dispatch!(level, simd => render_impl(simd, rect, strip_buf, alpha_buf));
}

/// Generates strip data for an axis-aligned rectangle.
///
/// # Strip layout strategy
///
/// Tile rows are classified into two kinds:
///
/// - **Edge rows** (top/bottom of rect): the rect boundary crosses partway
///   through the tile vertically, so individual pixels need per-cell alpha.
///   We emit a *single wide strip* spanning all tile columns, with alpha =
///   `x_alpha` * `y_alpha` (so the intersection of the alpha mask in each direction).
///
/// - **Interior rows**: every pixel in the tile has full vertical coverage,
///   so we only need to handle the left and right partial-column edges.
///   We emit a **left edge strip** (with its x-alpha mask) and, when the rect
///   spans more than one tile column, a **right edge strip** with `fill_gap =
///   true` so the renderer fills solid 0xFF between them.
///
/// The x-alpha masks for the left/right edge tiles are y-independent, so they
/// are precomputed once and reused across all interior rows.
#[inline(always)]
fn render_impl<S: Simd>(s: S, rect: Rect, strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u8>) {
    if rect.is_zero_area() {
        return;
    }

    let wide = rect.x1 > f64::from(u16::MAX) || rect.y1 > f64::from(u16::MAX);
    let coordinate = |value: f64| if wide { value } else { f64::from(value as f32) };
    let rect_x0 = coordinate(rect.x0);
    let rect_y0 = coordinate(rect.y0);
    let rect_x1 = coordinate(rect.x1);
    let rect_y1 = coordinate(rect.y1);

    // Integer pixel bounds.
    let px_x0 = rect_x0.floor() as u32;
    let px_y0 = rect_y0.floor() as u32;
    let px_y1 = rect_y1.ceil() as u32;

    let left_tile_x = (px_x0 / Tile::WIDTH_U32) * Tile::WIDTH_U32;
    // Inclusive, so don't use `ceil` here but just `rect_x1` directly.
    let right_tile_x = (rect_x1 as u32 / Tile::WIDTH_U32) * Tile::WIDTH_U32;

    let y0 = (px_y0 / Tile::HEIGHT_U32) * Tile::HEIGHT_U32;
    let y1 = px_y1
        .checked_next_multiple_of(Tile::HEIGHT_U32)
        .expect("source rectangle exceeds tile coordinate domain");
    // Include one tile past the right edge so the right-edge tile column is
    // covered by the edge-row wide-strip loop.
    let x_end = right_tile_x + Tile::WIDTH_U32;

    if x_end <= left_tile_x || y1 <= y0 {
        return;
    }

    let tile_start_y = y0 / Tile::HEIGHT_U32;
    let tile_end_y = y1 / Tile::HEIGHT_U32;

    // A right strip is only needed when the rect spans more than one tile column.
    let needs_right_strip = right_tile_x > left_tile_x;

    let left_x_cov = coverage(left_tile_x, rect_x0, rect_x1, wide);
    let right_x_cov = coverage(right_tile_x, rect_x0, rect_x1, wide);
    let left_x_mask = alpha_mask_from_x_coverage(s, &left_x_cov);
    let right_x_mask = alpha_mask_from_x_coverage(s, &right_x_cov);

    for tile_y in tile_start_y..tile_end_y {
        let strip_y = tile_y * Tile::HEIGHT_U32;
        let strip_y_f = f64::from(strip_y);
        let strip_y_end_f = f64::from(strip_y) + f64::from(Tile::HEIGHT_U32);

        // A row is an "edge" if the rect's top or bottom boundary falls
        // *inside* it (i.e. partial vertical coverage).
        let is_top_edge = strip_y_f < rect_y0 && rect_y0 < strip_y_end_f;
        let is_bottom_edge = strip_y_f < rect_y1 && rect_y1 < strip_y_end_f;

        if is_top_edge || is_bottom_edge {
            let alpha_start = Strip::alpha_index(alpha_buf.len());

            let y_cov = coverage(strip_y, rect_y0, rect_y1, wide);
            // Only the left-most and right-most tiles can have partial horizontal coverage,
            // all tiles in-between are fully covered horizontally.
            let left_alpha = combined_tile_alpha(s, &left_x_cov, &y_cov);
            let right_alpha = combined_tile_alpha(s, &right_x_cov, &y_cov);
            let interior_alpha = combined_tile_alpha(s, &[1.0; Tile::WIDTH_U32 as usize], &y_cov);
            let mut col = left_tile_x;
            while col + Tile::WIDTH_U32 <= x_end {
                let combined = if col == left_tile_x {
                    left_alpha
                } else if col == right_tile_x {
                    right_alpha
                } else {
                    interior_alpha
                };
                alpha_buf.extend_from_slice(combined.as_slice());
                col += Tile::WIDTH_U32;
            }

            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));
        } else {
            let alpha_start = Strip::alpha_index(alpha_buf.len());
            alpha_buf.extend_from_slice(left_x_mask.as_slice());
            strip_buf.push(Strip::new(left_tile_x, strip_y, alpha_start, false));

            if needs_right_strip {
                // `fill_gap = true` tells the renderer to fill solid 0xFF
                // between the previous strip's end and this strip's start.
                let alpha_start = Strip::alpha_index(alpha_buf.len());
                alpha_buf.extend_from_slice(right_x_mask.as_slice());
                strip_buf.push(Strip::new(right_tile_x, strip_y, alpha_start, true));
            }
        }
    }

    // Sentinel strip: marks the end of the strip list for this shape.
    let last_strip_y = (tile_end_y - 1) * Tile::HEIGHT_U32;
    strip_buf.push(Strip::sentinel(
        last_strip_y,
        Strip::alpha_index(alpha_buf.len()),
    ));
}

/// Compute fractional pixel coverage for `N` consecutive pixels starting at `start`.
#[inline(always)]
fn coverage<const N: usize>(start: u32, rect_lo: f64, rect_hi: f64, wide: bool) -> [f32; N] {
    let (start, rect_lo, rect_hi) = if wide {
        (
            0,
            (rect_lo - f64::from(start)) as f32,
            (rect_hi - f64::from(start)) as f32,
        )
    } else {
        (start, rect_lo as f32, rect_hi as f32)
    };
    let mut cov = [0.0_f32; N];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for i in 0..N {
        let px = (start as usize + i) as f32;
        cov[i] = (rect_hi.min(px + 1.0) - rect_lo.max(px)).clamp(0.0, 1.0);
    }
    cov
}

/// Build an alpha mask for the 4x4 tile from the given horizontal coverages,
/// splatting them across the other dimension.
#[inline(always)]
fn alpha_mask_from_x_coverage<S: Simd>(s: S, cov: &[f32; Tile::WIDTH_U32 as usize]) -> u8x16<S> {
    let mut buf = [0_u8; 16];

    #[allow(clippy::needless_range_loop, reason = "better clarity")]
    for col in 0..Tile::WIDTH_U32 as usize {
        let alpha = (cov[col] * 255.0 + 0.5) as u8;
        let base = col * Tile::HEIGHT_U32 as usize;
        buf[base..base + Tile::HEIGHT_U32 as usize].fill(alpha);
    }

    u8x16::from_slice(s, &buf)
}

/// Compute the alphas for a single 4x4 tile, taking horizontal as well as vertical coverage
/// of the rectangle into account.
#[inline(always)]
fn combined_tile_alpha<S: Simd>(
    s: S,
    x_cov: &[f32; Tile::WIDTH_U32 as usize],
    y_cov: &[f32; Tile::HEIGHT_U32 as usize],
) -> u8x16<S> {
    // Tiles are stored in column-major order, so each x coverage is repeated
    // for all rows, and the y coverages are repeated for all columns.
    let x_cov = element_wise_splat(s, f32x4::from_slice(s, x_cov));
    let y_cov = f32x16::block_splat(f32x4::from_slice(s, y_cov));

    f32_to_u8(x_cov * y_cov * 255.0 + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn translated_fractional_rect_keeps_coverage() {
        let mut reference_strips = Vec::new();
        let mut reference_alpha = Vec::new();
        let rect = Rect::new(1.0 / 256.0, 0.0, 1.0, 4.0);
        render(
            Level::baseline(),
            rect,
            &mut reference_strips,
            &mut reference_alpha,
        );
        assert_eq!(reference_alpha[0], 254);
        for dx in [70000_u32, 1 << 24] {
            let mut strips = Vec::new();
            let mut alpha = Vec::new();
            render(
                Level::baseline(),
                rect + crate::kurbo::Vec2::new(f64::from(dx), 0.0),
                &mut strips,
                &mut alpha,
            );
            for strip in &mut strips {
                if !strip.is_sentinel() {
                    strip.x -= dx;
                }
            }
            assert_eq!(strips, reference_strips, "dx={dx}");
            assert_eq!(alpha, reference_alpha, "dx={dx}");
        }
    }

    #[test]
    fn render_edge_row_at_u16_right_edge() {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        let rect = Rect::new(f64::from(65535_u32 - 3), 0.5, f64::from(65535_u32), 3.5);

        render(Level::baseline(), rect, &mut strips, &mut alphas);

        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].x, 65535_u32 - 3);
        assert_eq!(strips[0].alpha_idx(), 0);
        assert_eq!(
            alphas.len(),
            (Tile::WIDTH_U32 as usize) * (Tile::HEIGHT_U32 as usize)
        );
        assert!(strips[1].is_sentinel());
    }

    #[test]
    fn render_edge_row_at_u16_bottom_edge() {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        let rect = Rect::new(0.5, f64::from(65535_u32 - 3), 3.5, f64::from(65535_u32));

        render(Level::baseline(), rect, &mut strips, &mut alphas);

        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].y, 65535_u32 - 3);
        assert_eq!(strips[0].alpha_idx(), 0);
        assert_eq!(
            alphas.len(),
            (Tile::WIDTH_U32 as usize) * (Tile::HEIGHT_U32 as usize)
        );
        assert!(strips[1].is_sentinel());
    }
}
