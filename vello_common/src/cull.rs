// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Keeps the cost of stroking and dashing a segment bounded by the view rather than by
//! the size of its coordinates. (Fills clip their segments to the view in device
//! space instead: `fill_clip`.)
//!
//! Kurbo's `flatten` reserves memory for a cubic in proportion to the cube root of its
//! size and emits lines in proportion to the square root, and dashing walks every dash,
//! so one cubic with control points at 1e24 asks for gigabytes. Strokes and dashing
//! work in path space with the tolerance divided by the transform's largest stretch,
//! so their cost is bounded by a segment's path-space size times that stretch, which
//! an anisotropic transform can make arbitrarily larger than its device size. Segments
//! too large for their consumer are therefore split (de Casteljau, at the middle)
//! until each piece is no larger than the view or its control box misses the view; a
//! piece in view that is still too large in path space (the transform compresses it)
//! is drawn as a polyline through points of the curve, spaced by its device size. Real drawings stay
//! below these sizes, so their segments pass through unchanged.

use crate::flatten::TOL;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Affine, CubicBez, Line, ParamCurve, PathEl, PathSeg, Point, QuadBez};
use alloc::vec::Vec;

/// Segments whose device control box is wider or taller than this many pixels are
/// split, and so are curves whose path-space control box times the transform's largest
/// stretch is (for consumers in path space). A cubic this size flattens to a few
/// thousand lines at most.
pub(crate) const SPLIT_EXTENT: f64 = 131_072.0;

/// Halvings after which a piece that still crosses the view is drawn as its chord:
/// 2^128 pixels is beyond any `f32` coordinate under a sane transform.
const MAX_DEPTH: u32 = 128;

/// Splits of one segment after which its remaining pieces are drawn as chords.
const MAX_SPLITS: u32 = 4096;

/// Lines the pieces of one segment drawn as polylines have at most, together; a piece
/// no larger than a 65 535 px view needs about a thousand. Further pieces are chords.
const MAX_POLYLINE: u32 = 4096;

/// The view in device pixels, the transform from path to device space, and how large
/// a curve may be in path space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cull {
    affine: Affine,
    /// Left, top, right, bottom: geometry beyond them paints nothing.
    rect: [f64; 4],
    /// Curves whose path-space control box is wider or taller are too large for a
    /// consumer in path space; infinite for one in device space.
    path_limit: f64,
}

impl Cull {
    /// For a consumer in device space.
    #[cfg(test)]
    pub(crate) fn new(affine: Affine, rect: [f64; 4]) -> Self {
        Self {
            affine,
            rect,
            path_limit: f64::INFINITY,
        }
    }

    /// For a consumer in path space whose tolerance is the device tolerance divided by
    /// `scale`, the transform's largest stretch: its work on a curve is that of a
    /// device-space consumer on the curve scaled by `scale`.
    pub(crate) fn in_path_space(affine: Affine, rect: [f64; 4], scale: f64) -> Self {
        Self {
            affine,
            rect,
            path_limit: SPLIT_EXTENT / scale,
        }
    }

    /// The device bounding box of the control points of `seg` (left, top, right,
    /// bottom); NaN coordinates are left out.
    fn device_box(&self, seg: &PathSeg) -> [f64; 4] {
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut add = |p: Point| {
            let d = self.affine * p;
            b = [b[0].min(d.x), b[1].min(d.y), b[2].max(d.x), b[3].max(d.y)];
        };
        match *seg {
            PathSeg::Line(l) => [l.p0, l.p1].into_iter().for_each(&mut add),
            PathSeg::Quad(q) => [q.p0, q.p1, q.p2].into_iter().for_each(&mut add),
            PathSeg::Cubic(c) => [c.p0, c.p1, c.p2, c.p3].into_iter().for_each(&mut add),
        }
        b
    }

    fn misses(&self, b: [f64; 4]) -> bool {
        let r = self.rect;
        b[2] < r[0] || b[3] < r[1] || b[0] > r[2] || b[1] > r[3]
    }

    /// The size pieces of a split segment are reduced to: the view's larger side.
    fn piece_extent(&self) -> f64 {
        let r = self.rect;
        (r[2] - r[0]).max(r[3] - r[1]).max(1.0)
    }

    /// Whether `seg` is a curve too large in path space for its consumer.
    #[inline(always)]
    pub(crate) fn too_large(&self, seg: &PathSeg) -> bool {
        if self.path_limit == f64::INFINITY {
            return false;
        }
        let ext = |a: f64, b: f64, c: f64, d: f64| a.max(b).max(c).max(d) - a.min(b).min(c).min(d);
        let e = match *seg {
            PathSeg::Line(_) => return false,
            PathSeg::Quad(q) => {
                ext(q.p0.x, q.p1.x, q.p2.x, q.p2.x).max(ext(q.p0.y, q.p1.y, q.p2.y, q.p2.y))
            }
            PathSeg::Cubic(c) => {
                ext(c.p0.x, c.p1.x, c.p2.x, c.p3.x).max(ext(c.p0.y, c.p1.y, c.p2.y, c.p3.y))
            }
        };
        // NaN is not too large: the consumers drop paths with NaN.
        e > self.path_limit
    }

    /// The number of lines through evenly spaced parameters that keep `seg` within the
    /// flattening tolerance in device space: a chord over `1/n` of the parameter
    /// deviates by at most `max |c''| / (8 n²)`.
    fn polyline_lines(&self, seg: &PathSeg) -> u32 {
        let a = self.affine;
        let second = |p0: Point, p1: Point, p2: Point| {
            ((a * p0).to_vec2() - 2.0 * (a * p1).to_vec2() + (a * p2).to_vec2()).hypot()
        };
        // max |c''| is 2 |p0 - 2 p1 + p2| for a quadratic, and at most 6 times the
        // larger second difference for a cubic.
        let max_second = match *seg {
            PathSeg::Line(_) => 0.0,
            PathSeg::Quad(q) => 2.0 * second(q.p0, q.p1, q.p2),
            PathSeg::Cubic(c) => 6.0 * second(c.p0, c.p1, c.p2).max(second(c.p1, c.p2, c.p3)),
        };
        let n = (max_second / (8.0 * TOL)).sqrt().ceil();
        // NaN and infinity end up at the limit, with saturating casts.
        (n as u32).clamp(1, MAX_POLYLINE)
    }
}

fn exceeds(b: [f64; 4], extent: f64) -> bool {
    (b[2] - b[0]).max(b[3] - b[1]) > extent
}

/// A path whose segments too large for their consumer are split into pieces that are
/// each no larger than the view or outside it. Each element comes with whether it is
/// such a piece outside the view.
///
/// A piece outside the view becomes its chord, unless `keep_hidden` (dashing needs its
/// arc length; [`crate::dash::dash`] then skips it). The chord paints the same pixels
/// inside the view: for a fill, the region between a curve and its chord lies in the
/// control hull; for a stroke, `Cull`'s rect is widened by the outline's reach, which
/// also covers the joins at the piece's ends, whose angle the chord changes. Lines are
/// split only for dashing; a fill or an undashed stroke handles them in constant time.
/// For dashing, every segment larger than the view is split, so the dashes walked are
/// those in view.
pub(crate) struct SplitHuge<I> {
    inner: I,
    cull: Cull,
    keep_hidden: bool,
    /// Segments whose device control box is no larger pass through.
    pass_extent: f64,
    start: Point,
    last: Point,
    /// Pieces still to emit with their depth, the next one on top.
    stack: Vec<(PathSeg, u32)>,
    splits: u32,
    /// A piece being emitted as a polyline: the lines out so far and their number.
    polyline: Option<(PathSeg, u32, u32)>,
    /// Polyline lines the current segment may still have.
    polyline_left: u32,
    /// A `ClosePath` to emit once the closing line's pieces are out.
    close_pending: bool,
}

impl<I: Iterator<Item = PathEl>> SplitHuge<I> {
    pub(crate) fn new(
        path: impl IntoIterator<IntoIter = I>,
        cull: Cull,
        keep_hidden: bool,
    ) -> Self {
        Self {
            inner: path.into_iter(),
            cull,
            keep_hidden,
            pass_extent: if keep_hidden {
                cull.piece_extent()
            } else {
                SPLIT_EXTENT
            },
            start: Point::ZERO,
            last: Point::ZERO,
            stack: Vec::new(),
            splits: 0,
            polyline: None,
            polyline_left: MAX_POLYLINE,
            close_pending: false,
        }
    }

    /// The element for `seg` (a segment at depth 0, else a piece of one), or `None`
    /// once its halves are on the stack or its polyline has begun.
    fn piece(&mut self, seg: PathSeg, depth: u32) -> Option<(PathEl, bool)> {
        let b = self.cull.device_box(&seg);
        if depth == 0 && !exceeds(b, self.pass_extent) && !self.cull.too_large(&seg) {
            return Some((seg_to_el(&seg), false));
        }
        if self.cull.misses(b) {
            return Some(if self.keep_hidden {
                (seg_to_el(&seg), true)
            } else {
                (PathEl::LineTo(seg.end()), false)
            });
        }
        if !exceeds(b, self.cull.piece_extent()) {
            if !self.cull.too_large(&seg) {
                return Some((seg_to_el(&seg), false));
            }
            // Once the segment's lines are spent, its further pieces are chords: a
            // curve too large in path space would cost its consumer that size.
            let n = self
                .cull
                .polyline_lines(&seg)
                .min(self.polyline_left)
                .max(1);
            self.polyline_left = self.polyline_left.saturating_sub(n);
            self.polyline = Some((seg, 0, n));
            return None;
        }
        if depth >= MAX_DEPTH || self.splits >= MAX_SPLITS {
            return Some((PathEl::LineTo(seg.end()), false));
        }
        self.splits += 1;
        let (a, b) = halves(&seg);
        self.stack.push((b, depth + 1));
        self.stack.push((a, depth + 1));
        None
    }
}

impl<I: Iterator<Item = PathEl>> Iterator for SplitHuge<I> {
    type Item = (PathEl, bool);

    fn next(&mut self) -> Option<(PathEl, bool)> {
        loop {
            if let Some((seg, done, n)) = &mut self.polyline {
                *done += 1;
                let p = if *done == *n {
                    let end = seg.end();
                    self.polyline = None;
                    end
                } else {
                    seg.eval(f64::from(*done) / f64::from(*n))
                };
                return Some((PathEl::LineTo(p), false));
            }
            if let Some((seg, depth)) = self.stack.pop() {
                match self.piece(seg, depth) {
                    Some(el) => return Some(el),
                    None => continue,
                }
            }
            if self.close_pending {
                self.close_pending = false;
                return Some((PathEl::ClosePath, false));
            }
            let el = self.inner.next()?;
            let p0 = self.last;
            let seg = match el {
                PathEl::MoveTo(p) => {
                    self.start = p;
                    self.last = p;
                    return Some((el, false));
                }
                PathEl::LineTo(p) => {
                    self.last = p;
                    if !self.keep_hidden {
                        return Some((el, false));
                    }
                    PathSeg::Line(Line::new(p0, p))
                }
                PathEl::QuadTo(p1, p2) => {
                    self.last = p2;
                    PathSeg::Quad(QuadBez::new(p0, p1, p2))
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    self.last = p3;
                    PathSeg::Cubic(CubicBez::new(p0, p1, p2, p3))
                }
                PathEl::ClosePath => {
                    self.last = self.start;
                    let closing = PathSeg::Line(Line::new(p0, self.start));
                    if !self.keep_hidden
                        || p0 == self.start
                        || !exceeds(self.cull.device_box(&closing), self.pass_extent)
                    {
                        return Some((el, false));
                    }
                    // Dashing walks the closing line too: split it as an explicit line.
                    self.close_pending = true;
                    closing
                }
            };
            self.splits = 0;
            self.polyline_left = MAX_POLYLINE;
            if let Some(el) = self.piece(seg, 0) {
                return Some(el);
            }
        }
    }
}

/// `kurbo::flatten` (kurbo 0.13.1 `bezpath.rs`, the same arithmetic, so the same
/// lines), except that a curve `cull` finds too large in path space is first split by
/// [`SplitHuge`]. Lines and curves of normal size take no extra step.
pub(crate) fn flatten(
    path: impl IntoIterator<Item = PathEl>,
    tolerance: f64,
    cull: Cull,
    mut callback: impl FnMut(PathEl),
) {
    let sqrt_tol = tolerance.sqrt();
    let mut last_pt = None;
    let mut quad_buf = Vec::new();
    for el in path {
        match el {
            PathEl::MoveTo(p) => {
                last_pt = Some(p);
                callback(PathEl::MoveTo(p));
            }
            PathEl::LineTo(p) => {
                last_pt = Some(p);
                callback(PathEl::LineTo(p));
            }
            PathEl::QuadTo(p1, p2) => {
                if let Some(p0) = last_pt {
                    let q = QuadBez::new(p0, p1, p2);
                    if cull.too_large(&PathSeg::Quad(q)) {
                        flatten_split(
                            PathSeg::Quad(q),
                            tolerance,
                            cull,
                            &mut quad_buf,
                            &mut callback,
                        );
                    } else {
                        flatten_quad(q, sqrt_tol, &mut callback);
                    }
                }
                last_pt = Some(p2);
            }
            PathEl::CurveTo(p1, p2, p3) => {
                if let Some(p0) = last_pt {
                    let c = CubicBez::new(p0, p1, p2, p3);
                    if cull.too_large(&PathSeg::Cubic(c)) {
                        flatten_split(
                            PathSeg::Cubic(c),
                            tolerance,
                            cull,
                            &mut quad_buf,
                            &mut callback,
                        );
                    } else {
                        flatten_cubic(c, tolerance, sqrt_tol, &mut quad_buf, &mut callback);
                    }
                }
                last_pt = Some(p3);
            }
            PathEl::ClosePath => {
                last_pt = None;
                callback(PathEl::ClosePath);
            }
        }
    }
}

/// Flattens the pieces `SplitHuge` cuts `seg` into; each curve among them is small
/// enough in path space.
#[cold]
fn flatten_split(
    seg: PathSeg,
    tolerance: f64,
    cull: Cull,
    quad_buf: &mut Vec<(QuadBez, crate::flatten_simd::FlattenParams)>,
    callback: &mut impl FnMut(PathEl),
) {
    let sqrt_tol = tolerance.sqrt();
    let mut last = seg.start();
    let pieces = SplitHuge::new([PathEl::MoveTo(last), seg_to_el(&seg)], cull, false);
    for (el, _) in pieces.skip(1) {
        match el {
            PathEl::QuadTo(p1, p2) => flatten_quad(QuadBez::new(last, p1, p2), sqrt_tol, callback),
            PathEl::CurveTo(p1, p2, p3) => {
                flatten_cubic(
                    CubicBez::new(last, p1, p2, p3),
                    tolerance,
                    sqrt_tol,
                    quad_buf,
                    callback,
                );
            }
            el => callback(el),
        }
        last = el.end_point().unwrap_or(last);
    }
}

#[inline(always)]
fn flatten_quad(q: QuadBez, sqrt_tol: f64, callback: &mut impl FnMut(PathEl)) {
    use crate::flatten_simd::FlattenParamsExt as _;
    let params = q.estimate_subdiv(sqrt_tol);
    let n = ((0.5 * params.val / sqrt_tol).ceil() as usize).max(1);
    let step = 1.0 / (n as f64);
    for i in 1..n {
        let u = (i as f64) * step;
        let t = q.determine_subdiv_t(&params, u);
        let p = q.eval(t);
        callback(PathEl::LineTo(p));
    }
    callback(PathEl::LineTo(q.p2));
}

/// Kurbo's `TO_QUAD_TOL`: the share of the tolerance for approximating a cubic by
/// quadratics.
const TO_QUAD_TOL: f64 = 0.1;

#[inline(always)]
fn flatten_cubic(
    c: CubicBez,
    tolerance: f64,
    sqrt_tol: f64,
    quad_buf: &mut Vec<(QuadBez, crate::flatten_simd::FlattenParams)>,
    callback: &mut impl FnMut(PathEl),
) {
    use crate::flatten_simd::FlattenParamsExt as _;
    // Subdivide into quadratics, and estimate the number of subdivisions required for
    // each, summing to arrive at an estimate for the number of subdivisions for the
    // cubic. Also retain these parameters for later.
    let iter = c.to_quads(tolerance * TO_QUAD_TOL);
    quad_buf.clear();
    quad_buf.reserve(iter.size_hint().0);
    let sqrt_remain_tol = sqrt_tol * (1.0 - TO_QUAD_TOL).sqrt();
    let mut sum = 0.0;
    for (_, _, q) in iter {
        let params = q.estimate_subdiv(sqrt_remain_tol);
        sum += params.val;
        quad_buf.push((q, params));
    }
    let n = ((0.5 * sum / sqrt_remain_tol).ceil() as usize).max(1);

    // Iterate through the quadratics, outputting the points of subdivisions that fall
    // within that quadratic.
    let step = sum / (n as f64);
    let mut i = 1;
    let mut val_sum = 0.0;
    for (q, params) in quad_buf.iter() {
        let mut target = (i as f64) * step;
        let recip_val = params.val.recip();
        while target < val_sum + params.val {
            let u = (target - val_sum) * recip_val;
            let t = q.determine_subdiv_t(params, u);
            let p = q.eval(t);
            callback(PathEl::LineTo(p));
            i += 1;
            if i == n + 1 {
                break;
            }
            target = (i as f64) * step;
        }
        val_sum += params.val;
    }
    callback(PathEl::LineTo(c.p3));
}

/// `seg` split at its middle; the halves share the midpoint and keep the end points.
fn halves(seg: &PathSeg) -> (PathSeg, PathSeg) {
    match *seg {
        PathSeg::Line(l) => {
            let m = l.p0.midpoint(l.p1);
            (
                PathSeg::Line(Line::new(l.p0, m)),
                PathSeg::Line(Line::new(m, l.p1)),
            )
        }
        PathSeg::Quad(q) => {
            let (a, b) = q.subdivide();
            (PathSeg::Quad(a), PathSeg::Quad(b))
        }
        PathSeg::Cubic(c) => {
            let (a, b) = c.subdivide();
            (PathSeg::Cubic(a), PathSeg::Cubic(b))
        }
    }
}

pub(crate) fn seg_to_el(seg: &PathSeg) -> PathEl {
    match *seg {
        PathSeg::Line(l) => PathEl::LineTo(l.p1),
        PathSeg::Quad(q) => PathEl::QuadTo(q.p1, q.p2),
        PathSeg::Cubic(c) => PathEl::CurveTo(c.p1, c.p2, c.p3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::{BezPath, Line as KLine, ParamCurveNearest};

    const VIEW: [f64; 4] = [0.0, 0.0, 100.0, 100.0];

    fn split(path: &BezPath, affine: Affine, keep_hidden: bool) -> Vec<PathEl> {
        let split = SplitHuge::new(path.iter(), Cull::new(affine, VIEW), keep_hidden);
        split.map(|(el, _)| el).collect()
    }

    /// The cubic of the review: collinear, control points at ±1e24.
    fn huge_cubic() -> BezPath {
        let mut p = BezPath::new();
        p.move_to((10.0, 10.0));
        p.curve_to((1e24, 10.0), (-1e24, 10.0), (20.0, 10.0));
        p
    }

    #[test]
    fn segments_below_the_split_extent_pass_through() {
        let mut p = BezPath::new();
        p.move_to((-50_000.0, 5.0));
        p.quad_to((60_000.0, -60_000.0), (3.0, 4.0));
        p.curve_to((-60_000.0, 0.0), (0.0, 60_000.0), (10.0, 10.0));
        p.close_path();
        p.curve_to((20.0, 20.0), (30.0, 30.0), (f64::NAN, 1.0));
        p.move_to((500_000.0, 500_000.0));
        p.curve_to(
            (500_100.0, 400_000.0),
            (400_000.0, 500_000.0),
            (500_050.0, 500_050.0),
        );
        p.move_to((0.0, 0.0));
        p.line_to((1e9, 7.0));
        // Debug text, so the NaN compares equal.
        let same = |a: &[PathEl], b: &[PathEl]| {
            assert_eq!(alloc::format!("{a:?}"), alloc::format!("{b:?}"));
        };
        same(&split(&p, Affine::IDENTITY, false), p.elements());
        // Dashing splits every segment larger than the view: in a view 1.2e6 wide,
        // the 1e9-long line and nothing else.
        let wide = Cull::new(Affine::IDENTITY, [-6e5, -6e5, 6e5, 6e5]);
        let out: Vec<_> = SplitHuge::new(p.iter(), wide, true)
            .map(|(el, _)| el)
            .collect();
        same(&out[..8], &p.elements()[..8]);
        assert!(out.len() > p.elements().len());
        check_pieces(&out[8..], Point::ZERO, Point::new(1e9, 7.0), true);
        // In a view 100 wide, the curves too.
        let out = split(&p, Affine::IDENTITY, true);
        assert!(out.len() > p.elements().len() + 3);
    }

    /// Every piece starts where the previous one ended and the last ends at the
    /// segment's end; no curve piece is larger than the view unless it lies outside it
    /// (kept for dashing).
    fn check_pieces(out: &[PathEl], start: Point, end: Point, keep_hidden: bool) {
        let cull = Cull::new(Affine::IDENTITY, VIEW);
        let mut last = start;
        for el in out {
            let seg = match *el {
                PathEl::LineTo(p) => PathSeg::Line(KLine::new(last, p)),
                PathEl::QuadTo(p1, p2) => PathSeg::Quad(QuadBez::new(last, p1, p2)),
                PathEl::CurveTo(p1, p2, p3) => PathSeg::Cubic(CubicBez::new(last, p1, p2, p3)),
                _ => panic!("{el:?}"),
            };
            if !matches!(seg, PathSeg::Line(_)) {
                let b = cull.device_box(&seg);
                assert!(
                    !exceeds(b, 100.0) || (keep_hidden && cull.misses(b)),
                    "{seg:?}"
                );
            }
            last = seg.end();
        }
        assert_eq!(last, end);
    }

    #[test]
    fn a_huge_cubic_becomes_few_bounded_pieces() {
        let path = huge_cubic();
        for keep in [false, true] {
            let out = split(&path, Affine::IDENTITY, keep);
            assert_eq!(out[0], PathEl::MoveTo(Point::new(10.0, 10.0)));
            assert!(out.len() < 1000, "{} pieces", out.len());
            check_pieces(
                &out[1..],
                Point::new(10.0, 10.0),
                Point::new(20.0, 10.0),
                keep,
            );
        }
        // The largest finite f32 coordinates, and a quadratic.
        let m = f64::from(f32::MAX);
        let mut p = BezPath::new();
        p.move_to((10.0, 10.0));
        p.curve_to((m, -m), (-m, m), (20.0, 90.0));
        p.quad_to((-m, -m), (50.0, 50.0));
        let out = split(&p, Affine::IDENTITY, false);
        assert!(out.len() < 4000, "{} pieces", out.len());
    }

    /// Where a huge cubic crosses the view, the flattened pieces stay within the
    /// flattening tolerance of it.
    #[test]
    fn split_pieces_trace_the_curve_in_view() {
        // c(0.5) = (50, 50): the curve swings out by 1e7 and passes through the view.
        let third = 400.0 / 3.0;
        let c = CubicBez::new(
            (-1e7, -1e7),
            (1e7, -3e6),
            (-1e7 + third, 3e6 + third),
            (1e7, 1e7),
        );
        let mut path = BezPath::new();
        path.move_to(c.p0);
        path.curve_to(c.p1, c.p2, c.p3);
        let tolerance = 0.25;
        let mut lines = Vec::new();
        let mut last = c.p0;
        crate::kurbo::flatten(split(&path, Affine::IDENTITY, false), tolerance, |el| {
            if let PathEl::LineTo(p) = el {
                lines.push(KLine::new(last, p));
                last = p;
            }
        });
        assert!(lines.len() < 5000, "{} lines", lines.len());
        let near: Vec<_> = lines
            .iter()
            .filter(|l| l.p0.x.min(l.p1.x) < 110.0 && l.p0.x.max(l.p1.x) > -10.0)
            .collect();
        let mut inside = 0;
        for i in 0..=4000 {
            let p = c.eval(0.5 + (f64::from(i) - 2000.0) * 5e-9);
            if !(0.0..=100.0).contains(&p.x) || !(0.0..=100.0).contains(&p.y) {
                continue;
            }
            inside += 1;
            let d = near
                .iter()
                .map(|l| l.nearest(p, 1e-9).distance_sq)
                .fold(f64::INFINITY, f64::min);
            assert!(d.sqrt() <= tolerance + 1e-3, "{p:?} is {} away", d.sqrt());
        }
        assert!(inside > 100, "{inside} samples in view");
    }

    /// Thin strokes flatten with `flatten`: the lines kurbo's `flatten` gives, for
    /// curves of normal size.
    #[test]
    fn flatten_is_kurbos_for_curves_of_normal_size() {
        let mut p = BezPath::new();
        p.move_to((10.0, 10.0));
        p.quad_to((60.0, -20.0), (80.0, 40.0));
        p.curve_to((120.0, 90.0), (-30.0, 70.0), (20.0, 30.0));
        p.close_path();
        p.move_to((20.0, 30.0));
        p.curve_to((20.0, 30.0), (25.0, 30.0), (30.0, 30.0));
        p.line_to((-500.0, 900.0));
        p.curve_to((3000.0, -2000.0), (-2000.0, 2500.0), (40.0, 40.0));
        for affine in [
            Affine::IDENTITY,
            Affine::scale(3.7),
            Affine::scale_non_uniform(0.3, 5.0),
        ] {
            let scale = crate::flatten::max_scale(affine);
            let tolerance = TOL / scale;
            let (mut a, mut b) = (Vec::new(), Vec::new());
            flatten(
                p.iter(),
                tolerance,
                Cull::in_path_space(affine, VIEW, scale),
                |el| {
                    a.push(el);
                },
            );
            crate::kurbo::flatten(p.iter(), tolerance, |el| b.push(el));
            assert_eq!(a, b, "{affine:?}");
        }
    }

    /// A curve that a transform compresses 1e20-fold: huge in path space, it is cut
    /// to the view in device space and its pieces there drawn as polylines within the
    /// tolerance of the curve.
    #[test]
    fn a_squashed_curve_is_traced_within_the_tolerance() {
        // On the device, as in `split_pieces_trace_the_curve_in_view`.
        let third = 400.0 / 3.0;
        let c = CubicBez::new(
            (-1e7, -1e7),
            (1e7, -3e6),
            (-1e7 + third, 3e6 + third),
            (1e7, 1e7),
        );
        let affine = Affine::scale_non_uniform(1e-20, 1.0);
        let user = |p: Point| Point::new(p.x * 1e20, p.y);
        let mut path = BezPath::new();
        path.move_to(user(c.p0));
        path.curve_to(user(c.p1), user(c.p2), user(c.p3));
        let cull = Cull::in_path_space(affine, VIEW, 1.0);
        assert!(cull.too_large(&path.segments().next().unwrap()));
        let mut lines = Vec::new();
        let mut last = c.p0;
        flatten(path.iter(), TOL, cull, |el| {
            if let PathEl::LineTo(p) = el {
                let p = affine * p;
                lines.push(KLine::new(last, p));
                last = p;
            }
        });
        assert!(lines.len() < 5000, "{} lines", lines.len());
        assert_eq!(last, affine * user(c.p3));
        let near: Vec<_> = lines
            .iter()
            .filter(|l| l.p0.x.min(l.p1.x) < 110.0 && l.p0.x.max(l.p1.x) > -10.0)
            .collect();
        let mut inside = 0;
        for i in 0..=4000 {
            let p = c.eval(0.5 + (f64::from(i) - 2000.0) * 5e-9);
            if !(0.0..=100.0).contains(&p.x) || !(0.0..=100.0).contains(&p.y) {
                continue;
            }
            inside += 1;
            let d = near
                .iter()
                .map(|l| l.nearest(p, 1e-9).distance_sq)
                .fold(f64::INFINITY, f64::min);
            assert!(d.sqrt() <= TOL + 1e-3, "{p:?} is {} away", d.sqrt());
        }
        assert!(inside > 100, "{inside} samples in view");
        // Every line near the view stays within the tolerance of the curve.
        for l in &near {
            let m = l.p0.midpoint(l.p1);
            if (0.0..=100.0).contains(&m.x) && (0.0..=100.0).contains(&m.y) {
                assert!(
                    c.nearest(m, 1e-12).distance_sq.sqrt() <= TOL + 1e-3,
                    "{l:?}"
                );
            }
        }
    }

    /// The reviews' cases: a miter limit that widens the stroke's cull rect so much
    /// that the first piece in it spends the segment's polyline lines. Each piece
    /// after that must still be no larger in path space than the consumer allows.
    #[test]
    fn spent_polyline_lines_leave_chords() {
        // Half the width times the miter limit, as `stroke` widens the view.
        let widened = |reach: f64| [-reach, -reach, 100.0 + reach, 100.0 + reach];
        let mut squashed = BezPath::new();
        squashed.move_to((1e21, 10.0));
        squashed.curve_to((1e30, 10.0), (-1e30, 10.0), (2e21, 10.0));
        let mut swung = BezPath::new();
        swung.move_to((10.0, 10.0));
        swung.curve_to((1e24, 0.0), (-1e24, 1e24), (20.0, 10.0));
        let cases = [
            (
                squashed,
                Affine::scale_non_uniform(1e-20, 1.0),
                0.5 * 1.0 * 2e9,
            ),
            (swung, Affine::IDENTITY, 0.5 * 0.5 * 1.1e24),
        ];
        for (path, affine, reach) in cases {
            let scale = crate::flatten::max_scale(affine);
            let cull = Cull::in_path_space(affine, widened(reach + 1.0), scale);
            for keep in [false, true] {
                let mut last = Point::ZERO;
                let mut lines = 0;
                for (el, hidden) in SplitHuge::new(path.iter(), cull, keep) {
                    let seg = match el {
                        PathEl::MoveTo(p) => {
                            last = p;
                            continue;
                        }
                        PathEl::LineTo(p) => PathSeg::Line(KLine::new(last, p)),
                        PathEl::QuadTo(p1, p2) => PathSeg::Quad(QuadBez::new(last, p1, p2)),
                        PathEl::CurveTo(p1, p2, p3) => {
                            PathSeg::Cubic(CubicBez::new(last, p1, p2, p3))
                        }
                        PathEl::ClosePath => continue,
                    };
                    assert!(
                        hidden || !cull.too_large(&seg),
                        "{affine:?} {keep}: {seg:?}"
                    );
                    lines += usize::from(matches!(seg, PathSeg::Line(_)));
                    last = seg.end();
                }
                assert!(lines > 1000, "{affine:?} {keep}: {lines} lines");
            }
        }
    }
}
