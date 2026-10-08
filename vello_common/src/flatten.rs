// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flattening filled and stroked paths.

use crate::flatten_simd::{Callback, LinePathEl};
use crate::geometry::RectU16;
use crate::kurbo::{self, Affine, PathEl, Stroke, StrokeCtx, StrokeOpts};
use alloc::vec::Vec;
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
    line_buf.clear();
    let mut lb = FlattenerCallback {
        line_buf,
        start: Point::ZERO,
        p0: Point::ZERO,
        is_nan: false,
    };

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
    let scale = max_scale(affine);
    if scale.is_nan() || scale <= 0.0 {
        line_buf.clear();
        return;
    }
    if style.width * scale <= HAIRLINE_MAX_WIDTH {
        hairline(path, style, affine, scale, line_buf, flatten_ctx, cull_bbox);
        return;
    }
    // The tolerance is in user space, so it shrinks with the largest stretch of the
    // transform (rotated transforms included).
    let tolerance = TOL / scale.max(1.);

    expand_stroke(path, style, tolerance, stroke_ctx);
    fill(
        level,
        stroke_ctx.output(),
        affine,
        line_buf,
        flatten_ctx,
        cull_bbox,
    );
}

/// Strokes at most this wide in device pixels skip stroke expansion: each flattened
/// segment becomes its own rectangle. Joins and caps then differ from the exact
/// outline by at most half the width, which is below a pixel.
pub const HAIRLINE_MAX_WIDTH: f64 = 1.0;

/// The largest factor by which `affine` stretches any direction (its largest singular
/// value).
pub fn max_scale(affine: Affine) -> f64 {
    let [a, b, c, d, _, _] = affine.as_coeffs();
    let t = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    (0.5 * (t + (t * t - 4.0 * det * det).max(0.0).sqrt())).sqrt()
}

/// Outlines a thin stroke with straight lines, built in user space and transformed, so
/// the geometry stays exact under any affine transform; it replaces stroke expansion,
/// whose curved round caps and joins dominate the cost of hairlines.
///
/// Each subpath becomes one polygon: the left offsets forward, the right offsets back
/// (a closed subpath gives two loops), with the same orientation whatever the
/// direction, as kurbo's outline has. At a join the inner side goes through the
/// intersection of the two offset lines, so the overlap is not counted twice, unless
/// that point lies beyond one of the segments (then it goes through the join point, as
/// kurbo does); the outer side takes the miter tip within the miter limit, else the
/// bevel, and round joins add the arc's midpoint. Joins whose wedge is under
/// `MIN_JOIN_AREA` take the miter tip whatever the style.
///
/// Open ends are extended by the cap: half the width for square caps, a quarter of it
/// for round caps. A quarter matches the area the expanded round cap keeps once
/// flattened at sub-pixel radius (two chords, `r²`), so both paths agree at the
/// threshold; the exact half disc (`π/8`) made stipple patterns of short round-capped
/// strokes visibly darker than other renderers. Degenerate subpaths draw nothing, as in
/// kurbo's stroker. A subpath outside `cull_bbox` is closed, so it contributes no
/// winding inside it and is dropped.
fn hairline(
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    affine: Affine,
    scale: f64,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
    cull_bbox: RectU16,
) {
    line_buf.clear();
    let [left, right] = &mut flatten_ctx.hairline_sides;
    left.clear();
    right.clear();
    let mut sink = HairlineSink {
        line_buf,
        left,
        right,
        affine,
        half_width: 0.5 * style.width,
        device_half_width2: (0.5 * style.width * scale).powi(2),
        join: style.join,
        miter_limit: style.miter_limit,
        start_ext: cap_extension(style.start_cap, style.width),
        end_ext: cap_extension(style.end_cap, style.width),
        cull: (
            f64::from(cull_bbox.x0),
            f64::from(cull_bbox.y0),
            f64::from(cull_bbox.x1),
            f64::from(cull_bbox.y1),
        ),
        start: kurbo::Point::ZERO,
        last: kurbo::Point::ZERO,
        first: None,
        prev: None,
        is_nan: false,
    };
    let tolerance = TOL / scale;
    if style.dash_pattern.is_empty() {
        kurbo::flatten(path, tolerance, |el| sink.push(el));
    } else {
        let dashed = kurbo::dash(path.into_iter(), style.dash_offset, &style.dash_pattern);
        kurbo::flatten(dashed, tolerance, |el| sink.push(el));
    }
    sink.end_subpath(false);
    if sink.is_nan {
        warn!("A path contains NaN, ignoring it.");
        sink.line_buf.clear();
    }
}

/// Joins whose wedge is smaller than this many square device pixels take the miter
/// tip whatever the join style (see `hairline`).
const MIN_JOIN_AREA: f64 = 1.0 / 128.0;

fn cap_extension(cap: kurbo::Cap, width: f64) -> f64 {
    match cap {
        kurbo::Cap::Butt => 0.0,
        kurbo::Cap::Square => 0.5 * width,
        kurbo::Cap::Round => 0.25 * width,
    }
}

/// A segment's unit direction and length.
#[derive(Clone, Copy)]
struct Dir {
    u: kurbo::Vec2,
    len: f64,
}

struct HairlineSink<'a> {
    line_buf: &'a mut Vec<Line>,
    /// The outline's points left and right of the subpath, in user space.
    left: &'a mut Vec<kurbo::Point>,
    right: &'a mut Vec<kurbo::Point>,
    affine: Affine,
    half_width: f64,
    /// The squared half width in device pixels.
    device_half_width2: f64,
    join: kurbo::Join,
    miter_limit: f64,
    start_ext: f64,
    end_ext: f64,
    /// Left, top, right, bottom in device pixels.
    cull: (f64, f64, f64, f64),
    start: kurbo::Point,
    last: kurbo::Point,
    /// The subpath's first and latest segment.
    first: Option<Dir>,
    prev: Option<Dir>,
    is_nan: bool,
}

impl HairlineSink<'_> {
    fn push(&mut self, el: PathEl) {
        match el {
            PathEl::MoveTo(p) => {
                self.end_subpath(false);
                self.start = p;
                self.last = p;
            }
            PathEl::LineTo(p) => self.segment(p),
            PathEl::ClosePath => {
                if self.last != self.start {
                    self.segment(self.start);
                }
                self.end_subpath(true);
                self.last = self.start;
            }
            // `kurbo::flatten` emits only lines.
            PathEl::QuadTo(..) | PathEl::CurveTo(..) => unreachable!(),
        }
    }

    fn normal(&self, d: Dir) -> kurbo::Vec2 {
        kurbo::Vec2::new(-d.u.y, d.u.x) * self.half_width
    }

    fn segment(&mut self, p: kurbo::Point) {
        if p == self.last {
            return;
        }
        let v = p - self.last;
        let len = v.hypot();
        let d = Dir { u: v / len, len };
        match self.prev {
            None => {
                let n = self.normal(d);
                self.left.push(self.last + n);
                self.right.push(self.last - n);
                self.first = Some(d);
            }
            Some(prev) => self.join(self.last, prev, d),
        }
        self.prev = Some(d);
        self.last = p;
    }

    /// Adds the outline points of the join at `p` from `d0` into `d1` to both sides.
    fn join(&mut self, p: kurbo::Point, d0: Dir, d1: Dir) {
        let (n0, n1) = (self.normal(d0), self.normal(d1));
        let cos = d0.u.dot(d1.u);
        let turn = d0.u.cross(d1.u);
        let tan_half = turn.abs() / (1.0 + cos);
        // Where the two offset lines on either side meet, relative to `p`, on the left.
        let miter = (n0 + n1) / (1.0 + cos);
        // Turning left (positive cross product, y up) puts the left side inside.
        let (inner, outer) = if turn > 0.0 {
            (&mut *self.left, &mut *self.right)
        } else {
            (&mut *self.right, &mut *self.left)
        };
        let side = if turn > 0.0 { 1.0 } else { -1.0 };
        let (n0, n1, miter) = (n0 * side, n1 * side, miter * side);
        // The inner side, which `miter`, `n0` and `n1` point to.
        let reach = self.half_width * tan_half;
        if reach.is_finite() && reach <= d0.len.min(d1.len) {
            inner.push(p + miter);
        } else {
            inner.extend([p + n0, p, p + n1]);
        }
        // The outer side.
        let negligible = tan_half.is_nan() || self.device_half_width2 * tan_half < MIN_JOIN_AREA;
        let within_limit = (0.5 * (1.0 + cos)).sqrt() * self.miter_limit >= 1.0;
        match self.join {
            _ if negligible && reach.is_finite() => outer.push(p - miter),
            kurbo::Join::Miter if within_limit => outer.push(p - miter),
            kurbo::Join::Round => {
                let mid = n0 + n1;
                let mid_len = mid.hypot();
                // A full reversal bulges forward along the incoming segment.
                let tip = if mid_len > 0.0 {
                    p - mid * (self.half_width / mid_len)
                } else {
                    p + d0.u * self.half_width
                };
                outer.extend([p - n0, tip, p - n1]);
            }
            _ => outer.extend([p - n0, p - n1]),
        }
    }

    fn end_subpath(&mut self, closed: bool) {
        let (Some(first), Some(last)) = (self.first.take(), self.prev.take()) else {
            return;
        };
        if closed {
            // The join at the start replaces the first points pushed; the loops are
            // cyclic, so its points can follow the last join's.
            self.join(self.start, last, first);
            self.left.remove(0);
            self.right.remove(0);
            self.emit_loop(false);
            self.emit_loop(true);
        } else {
            let n = self.normal(last);
            let e0 = first.u * self.start_ext;
            let e1 = last.u * self.end_ext;
            self.left.push(self.last + n + e1);
            self.right.push(self.last - n + e1);
            self.left[0] -= e0;
            self.right[0] -= e0;
            self.emit_open();
        }
        self.left.clear();
        self.right.clear();
    }

    /// Emits the left side forward and the right side back as one polygon.
    fn emit_open(&mut self) {
        let mut pts = core::mem::take(self.left);
        pts.extend(self.right.iter().rev());
        self.emit(&pts);
        *self.left = pts;
    }

    /// Emits one side of a closed subpath as a loop: the left side forward, the right
    /// side backward.
    fn emit_loop(&mut self, right: bool) {
        let mut pts = core::mem::take(if right {
            &mut *self.right
        } else {
            &mut *self.left
        });
        if right {
            pts.reverse();
        }
        self.emit(&pts);
        if right {
            pts.reverse();
            *self.right = pts;
        } else {
            *self.left = pts;
        }
    }

    /// Emits a closed polygon given in user space, unless it lies outside the cull box.
    fn emit(&mut self, pts: &[kurbo::Point]) {
        let start = self.line_buf.len();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let mut prev = Point::from(self.affine * pts[pts.len() - 1]);
        for p in pts {
            let d = self.affine * *p;
            self.is_nan |= d.x.is_nan() || d.y.is_nan();
            x0 = x0.min(d.x);
            y0 = y0.min(d.y);
            x1 = x1.max(d.x);
            y1 = y1.max(d.y);
            let d = Point::from(d);
            self.line_buf.push(Line::new(prev, d));
            prev = d;
        }
        let (left, top, right, bottom) = self.cull;
        if x1 < left || y1 < top || x0 > right || y0 > bottom {
            self.line_buf.truncate(start);
        }
    }
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

struct FlattenerCallback<'a> {
    line_buf: &'a mut Vec<Line>,
    start: Point,
    p0: Point,
    is_nan: bool,
}

impl Callback for FlattenerCallback<'_> {
    #[inline(always)]
    fn callback(&mut self, el: LinePathEl) {
        match el {
            LinePathEl::MoveTo(p) => {
                self.is_nan |= p.is_nan();

                let p = p.into();
                self.start = p;
                self.p0 = p;
            }
            LinePathEl::LineTo(p) => {
                self.is_nan |= p.is_nan();

                let p = p.into();
                self.line_buf.push(Line::new(self.p0, p));
                self.p0 = p;
            }
        }
    }
}

#[cfg(test)]
mod hairline_tests {
    use super::*;

    fn lines(path: &kurbo::BezPath, style: &Stroke, affine: Affine, cull: RectU16) -> Vec<Line> {
        let mut buf = Vec::new();
        hairline(
            path.iter(),
            style,
            affine,
            max_scale(affine),
            &mut buf,
            &mut FlattenCtx::default(),
            cull,
        );
        buf
    }

    fn bounds(lines: &[Line]) -> (f32, f32, f32, f32) {
        lines.iter().flat_map(|l| [l.p0, l.p1]).fold(
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
            |(x0, y0, x1, y1), p| (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y)),
        )
    }

    const VIEW: RectU16 = RectU16 {
        x0: 0,
        y0: 0,
        x1: 100,
        y1: 100,
    };

    #[test]
    fn open_segment_gets_caps_at_both_ends() {
        let mut p = kurbo::BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((20.0, 10.0));
        let style = Stroke::new(0.5).with_caps(kurbo::Cap::Square);
        let l = lines(&p, &style, Affine::scale(2.0), VIEW);
        assert_eq!(l.len(), 4);
        // User width 0.5 at scale 2 is 1 px; square caps add half the width per end.
        assert_eq!(bounds(&l), (19.5, 19.5, 40.5, 20.5));
    }

    /// Signed areas of the closed polygons in `lines`, in emission order.
    fn polygon_areas(lines: &[Line]) -> Vec<f32> {
        let mut out = Vec::new();
        let (mut start, mut area) = (None, 0.0);
        for l in lines {
            let s = *start.get_or_insert(l.p0);
            area += l.p0.x * l.p1.y - l.p1.x * l.p0.y;
            if l.p1 == s {
                out.push(area);
                (start, area) = (None, 0.0);
            }
        }
        assert!(start.is_none(), "every polygon is closed");
        out
    }

    #[test]
    fn closed_subpath_is_two_loops_without_caps() {
        let mut p = kurbo::BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((20.0, 10.0));
        p.line_to((20.0, 20.0));
        p.line_to((10.0, 20.0));
        p.close_path();
        let style = Stroke::new(1.0)
            .with_caps(kurbo::Cap::Square)
            .with_join(kurbo::Join::Miter);
        let l = lines(&p, &style, Affine::IDENTITY, VIEW);
        let areas = polygon_areas(&l);
        assert_eq!(areas.len(), 2);
        // An 11 x 11 outer and a 9 x 9 inner square of opposite orientation: the
        // ring has the area of the stroke (shoelace sums are twice the area).
        let total: f32 = areas.iter().sum();
        assert!((total.abs() / 2.0 - 40.0).abs() < 1e-3, "{areas:?}");
        assert_eq!(bounds(&l), (9.5, 9.5, 20.5, 20.5));
    }

    #[test]
    fn miter_join_area_is_the_union() {
        let mut p = kurbo::BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((20.0, 10.0));
        p.line_to((20.0, 20.0));
        let style = Stroke::new(1.0)
            .with_caps(kurbo::Cap::Butt)
            .with_join(kurbo::Join::Miter);
        let l = lines(&p, &style, Affine::IDENTITY, VIEW);
        let areas = polygon_areas(&l);
        // Two 10 x 1 rectangles, plus the 0.5 x 0.5 square at the outer corner, minus
        // their 0.5 x 0.5 overlap at the inner one.
        assert_eq!(areas.len(), 1);
        assert!((areas[0].abs() / 2.0 - 20.0).abs() < 1e-3, "{areas:?}");
    }

    #[test]
    fn orientation_is_independent_of_direction_and_offscreen_subpaths_are_dropped() {
        let mut p = kurbo::BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((50.0, 10.0));
        p.line_to((10.0, 50.0));
        p.line_to((60.0, 60.0));
        p.move_to((90.0, 80.0));
        p.line_to((70.0, 80.0));
        p.line_to((80.0, 90.0));
        p.move_to((500.0, 10.0));
        p.line_to((600.0, 10.0));
        let flip = Affine::scale_non_uniform(1.0, -1.0) * Affine::translate((0.0, -100.0));
        for join in [kurbo::Join::Miter, kurbo::Join::Round, kurbo::Join::Bevel] {
            let l = lines(&p, &Stroke::new(1.0).with_join(join), flip, VIEW);
            let areas = polygon_areas(&l);
            assert_eq!(areas.len(), 2, "the third subpath lies outside the view");
            assert_eq!(areas[0].signum(), areas[1].signum(), "{join:?} {areas:?}");
        }
    }

    #[test]
    fn reversal_and_nan_are_handled() {
        let mut p = kurbo::BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((20.0, 10.0));
        p.line_to((10.0, 10.0));
        for join in [kurbo::Join::Miter, kurbo::Join::Round, kurbo::Join::Bevel] {
            let l = lines(
                &p,
                &Stroke::new(1.0).with_join(join),
                Affine::IDENTITY,
                VIEW,
            );
            assert!(
                l.iter().all(|l| !l.p0.x.is_nan() && !l.p0.y.is_nan()),
                "{join:?}"
            );
            assert!(!l.is_empty());
        }
        p.line_to((f64::NAN, 10.0));
        assert!(lines(&p, &Stroke::new(1.0), Affine::IDENTITY, VIEW).is_empty());
    }

    #[test]
    fn wide_strokes_are_expanded() {
        let mut p = kurbo::BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((20.0, 10.0));
        let mut buf = Vec::new();
        stroke(
            Level::new(),
            p.iter(),
            &Stroke::new(4.0),
            Affine::IDENTITY,
            &mut buf,
            &mut FlattenCtx::default(),
            &mut StrokeCtx::default(),
            VIEW,
        );
        // Round caps produce more than the four sides of a rectangle.
        assert!(buf.len() > 4);
    }
}
