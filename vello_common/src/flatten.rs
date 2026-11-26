// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flattening filled and stroked paths.

mod hairline;
use hairline::hairline;

use crate::cull::{Cull, SplitHuge};
use crate::flatten_simd::{Callback, LinePathEl};
use crate::geometry::RectU16;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{self, Affine, PathEl, Stroke, StrokeCtx, StrokeOpts};
use crate::path_budget::Meter;
use alloc::vec::Vec;
use core::f64::consts::SQRT_2;
use fearless_simd::{Level, Simd, dispatch};
use log::warn;

pub use crate::flatten_simd::FlattenCtx;

// The current tolerance is set to 0.25. Since `sqrt` doesn't work in const contexts, we instead
// hardcode the squared tolerance and derive the others from that.
pub(crate) const SQRT_TOL: f64 = 0.5;
pub(crate) const TOL: f64 = SQRT_TOL * SQRT_TOL;
pub(crate) const TOL_2: f64 = TOL * TOL;

/// A point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// The x coordinate of the point.
    pub x: f32,
    /// The y coordinate of the point.
    pub y: f32,
}

impl Point {
    /// The point `(0, 0)`.
    pub const ZERO: Self = Self::new(0., 0.);

    /// Create a new point.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl From<kurbo::Point> for Point {
    #[inline(always)]
    fn from(value: kurbo::Point) -> Self {
        Self {
            x: value.x as f32,
            y: value.y as f32,
        }
    }
}

impl core::ops::Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl core::ops::Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl core::ops::Mul<f32> for Point {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// A line.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    /// The start point of the line.
    pub p0: Point,
    /// The end point of the line.
    pub p1: Point,
}

impl Line {
    /// Create a new line.
    pub fn new(p0: Point, p1: Point) -> Self {
        Self { p0, p1 }
    }
}

/// Flatten a filled Bézier path into line segments.
///
/// # Open subpaths and culling
///
/// Open subpaths in the input path get closed by connecting the last endpoint in the subpath to
/// the starting point. The output lines in `line_buf` describe the flattened path, but these lines
/// may describe open subpaths, as some path elements may have been culled.
///
/// For example, consider the following, where the box describes the viewport, a path is marked by
/// `*`, and the region to be filled in the viewport is shaded. For ease of drawing the ASCII art,
/// the path elements are all lines ([`PathEl::LineTo`]), but the same also holds for Bézier path
/// elements.
///
/// ```text
///    ---> winding scan direction
///
///                   * * * * *
///                 *         *
///    ---------- * -----     *
///    |        *░░░░░░░|     *
///    |      *░░░░░░░░░|     *
///    |    *░░░░░░░░░░░|     *
///    |  *░░░░░░░░░░░░░|     *
///    |*░░░░░░░░░░░░░░░|     *
///   *|░░░░░░░░░░░░░░░░|     *
/// *  |░░░░░░░░░░░░░░░░|     *
/// *  |░░░░░░░░░░░░░░░░|     *
/// *  ------------------     *
/// *                         *
/// *                         *
/// *                         *
/// *                         *
/// *                         *
/// *                         *
/// *                         *
/// * * * * * * * * * * * * * *
/// ```
///
/// Because the winding scan direction is from left to right, only the left-of-viewport and
/// diagonal lines matter in later stages of rendering for the winding number and pixel coverage.
/// The other three lines can be culled.
///
/// ```text
///                   *
///                 *
///    ---------- * -----
///    |        *░░░░░░░|
///    |      *░░░░░░░░░|
///    |    *░░░░░░░░░░░|
///    |  *░░░░░░░░░░░░░|
///    |*░░░░░░░░░░░░░░░|
///   *|░░░░░░░░░░░░░░░░|
/// *  |░░░░░░░░░░░░░░░░|
/// *  |░░░░░░░░░░░░░░░░|
/// *  ------------------
/// *
/// *
/// *
/// *
/// *
/// *
/// *
/// ```
///
/// It is important to keep these flattened subpaths open after culling, as closing the subpaths
/// might yield different geometry like the following.
///
/// ```text
///                   *
///                 **
///    ---------- * *----
///    |        *░░*    |
///    |      *░░░*     |
///    |    *░░░░*      |
///    |  *░░░░░*       |
///    |*░░░░░░*        |
///   *|░░░░░░*         |
/// *  |░░░░░*          |
/// *  |░░░░*           |
/// *  --- * ------------
/// *     *
/// *    *
/// *   *
/// *  *
/// * *
/// **
/// *
/// ```
pub fn fill(
    level: Level,
    path: impl IntoIterator<Item = PathEl>,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    ctx: &mut FlattenCtx,
    cull_bbox: RectU16,
) {
    dispatch!(level, simd => fill_impl(simd, path, affine, line_buf, ctx, cull_bbox));
}

/// Flatten a filled bezier path into line segments.
///
/// See the note about open subpaths and culling on [`fill`].
#[inline(always)]
pub fn fill_impl<S: Simd>(
    simd: S,
    path: impl IntoIterator<Item = PathEl>,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
    cull_bbox: RectU16,
) {
    fill_path(
        simd,
        path,
        affine,
        line_buf,
        flatten_ctx,
        cull_bbox,
        Meter::default(),
    );
}

#[inline(always)]
pub(crate) fn fill_path<S: Simd>(
    simd: S,
    path: impl IntoIterator<Item = PathEl>,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
    cull_bbox: RectU16,
    meter: Meter<'_>,
) {
    line_buf.clear();
    let mut lb = FlattenerCallback::new(line_buf, meter, cull_bbox);
    crate::flatten_simd::flatten(simd, path, affine, &mut lb, flatten_ctx, cull_bbox);

    // A path that contains NaN is ill-defined, so ignore it.
    if lb.is_nan {
        warn!("A path contains NaN, ignoring it.");

        line_buf.clear();
    }
}

/// Flatten a stroked Bézier path into line segments.
///
/// See the note about open subpaths and culling on [`fill`].
pub fn stroke(
    level: Level,
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
    stroke_ctx: &mut StrokeCtx,
    cull_bbox: RectU16,
) {
    stroke_metered(
        level,
        path,
        style,
        affine,
        line_buf,
        flatten_ctx,
        stroke_ctx,
        cull_bbox,
        Meter::default(),
    );
}

pub(crate) fn stroke_metered(
    level: Level,
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
    stroke_ctx: &mut StrokeCtx,
    cull_bbox: RectU16,
    meter: Meter<'_>,
) {
    let scale = max_scale(affine);
    if scale.is_nan() || scale <= 0.0 {
        line_buf.clear();
        return;
    }
    // Thin strokes are flattened and dashes measured in path space, with a tolerance
    // divided by `scale`.
    let cull = Cull::in_path_space(affine, stroke_cull_rect(cull_bbox, style, scale), scale);
    let cull = cull.metered(meter);
    if style.width * scale <= HAIRLINE_MAX_WIDTH {
        hairline(
            path,
            style,
            affine,
            scale,
            line_buf,
            flatten_ctx,
            cull_bbox,
            cull,
            meter,
        );
        return;
    }
    // The tolerance is in user space, so it shrinks with the largest stretch of the
    // transform (rotated transforms included).
    let tolerance = TOL / scale.max(1.);

    // Kurbo's stroker does a bounded amount of work per segment (offset curves
    // subdivide at most `MAX_DEPTH` 8 times, kurbo 0.13.1 `offset.rs`) and `fill`
    // flattens its outline in device space, splitting huge curves itself: an undashed
    // stroke needs no splitting here.
    if style.dash_pattern.is_empty() {
        expand_stroke(path, style, tolerance, stroke_ctx);
    } else {
        let path = SplitHuge::new(path, cull, true);
        let mut dashed = crate::dash::dash(path, style.dash_offset, &style.dash_pattern, affine);
        dashed.meter = meter;
        let solid = Stroke {
            dash_pattern: kurbo::Dashes::new(),
            width: style.width,
            join: style.join,
            miter_limit: style.miter_limit,
            start_cap: style.start_cap,
            end_cap: style.end_cap,
            dash_offset: 0.0,
        };
        expand_stroke(dashed, &solid, tolerance, stroke_ctx);
    }
    dispatch!(level, simd => fill_path(simd, stroke_ctx.output().iter(), affine, line_buf, flatten_ctx, cull_bbox, meter));
}

/// Strokes at most this wide in device pixels skip stroke expansion and are outlined
/// with straight lines (see `hairline`). Their joins and caps then differ from the
/// exact outline by at most half the width, which is below a pixel.
const HAIRLINE_MAX_WIDTH: f64 = 1.0;

/// `cull_bbox` widened by how far a stroke's outline reaches from its path, in device
/// pixels: half the width times the miter limit for miter joins, else times √2 (square
/// caps; the hairline joins that take the miter tip turn by at most 90°), plus a pixel
/// for the flattening tolerance.
fn stroke_cull_rect(cull_bbox: RectU16, style: &Stroke, scale: f64) -> [f64; 4] {
    let miter = if style.join == kurbo::Join::Miter {
        style.miter_limit
    } else {
        0.0
    };
    let reach = 0.5 * style.width * scale * miter.max(SQRT_2) + 1.0;
    [
        f64::from(cull_bbox.x0) - reach,
        f64::from(cull_bbox.y0) - reach,
        f64::from(cull_bbox.x1) + reach,
        f64::from(cull_bbox.y1) + reach,
    ]
}

/// The largest factor by which `affine` stretches any direction (its largest singular
/// value).
pub(crate) fn max_scale(affine: Affine) -> f64 {
    let [a, b, c, d, _, _] = affine.as_coeffs();
    let t = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    (0.5 * (t + (t * t - 4.0 * det * det).max(0.0).sqrt())).sqrt()
}

/// Expand a stroked path to a filled path.
pub fn expand_stroke(
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    tolerance: f64,
    stroke_ctx: &mut StrokeCtx,
) {
    kurbo::stroke_with(path, style, &StrokeOpts::default(), tolerance, stroke_ctx);
}

/// The metered API exposes bounded edges to its caller. Raw flattening also
/// supports tile binning, which can consume ordinary offscreen endpoints directly.
fn emit_edge(
    lines: &mut Vec<Line>,
    meter: Meter<'_>,
    rect: [f64; 4],
    a: kurbo::Point,
    b: kurbo::Point,
) {
    let inside =
        |p: kurbo::Point| rect[0] <= p.x && p.x <= rect[2] && rect[1] <= p.y && p.y <= rect[3];
    let mut push = |p: kurbo::Point, q: kurbo::Point| {
        if meter.room(lines, 1) {
            lines.push(Line::new(p.into(), q.into()));
        }
    };
    if meter.0.is_some() && !(inside(a) && inside(b)) {
        if a.is_finite() && b.is_finite() {
            crate::fill_clip::directed_line(a, b, rect, push);
        }
    } else {
        push(a, b);
    }
}

struct FlattenerCallback<'a> {
    meter: Meter<'a>,
    line_buf: &'a mut Vec<Line>,
    rect: [f64; 4],
    p0: kurbo::Point,
    is_nan: bool,
}

impl<'a> FlattenerCallback<'a> {
    fn new(line_buf: &'a mut Vec<Line>, meter: Meter<'a>, bbox: RectU16) -> Self {
        Self {
            line_buf,
            meter,
            rect: [
                bbox.x0.into(),
                bbox.y0.into(),
                bbox.x1.into(),
                bbox.y1.into(),
            ],
            p0: kurbo::Point::ZERO,
            is_nan: false,
        }
    }
}

impl Callback for FlattenerCallback<'_> {
    fn meter(&self) -> Meter<'_> {
        self.meter
    }
    #[inline(always)]
    fn callback(&mut self, el: LinePathEl) {
        match el {
            LinePathEl::MoveTo(p) => {
                self.is_nan |= p.is_nan();

                self.p0 = p;
            }
            LinePathEl::LineTo(p) => {
                self.is_nan |= p.is_nan();

                emit_edge(self.line_buf, self.meter, self.rect, self.p0, p);
                self.p0 = p;
            }
        }
    }
}
