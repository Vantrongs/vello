// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clipping a fill's segments to the view in device space before flattening them.
//!
//! A segment of a filled path whose device control box is wider or taller than
//! `cull::SPLIT_EXTENT`, or a cubic too curved for the flattener's `MAX_QUADS`
//! quadratics, is not flattened whole: its work would grow with its size, and `f32`
//! lines that long lose its position in the view. Instead, in `f64` device space:
//!
//! 1. It is split at its x and y extrema into pieces monotone in both.
//! 2. Each piece is cut where it crosses the view's edges, found by bisection on the
//!    piece's parameter interval; the cut point lies exactly on the edge.
//! 3. Winding is counted along rows from the left, so what lies above, below or right
//!    of the view changes nothing in it and is dropped, and what lies left of it is
//!    replaced by a vertical line on the view's left edge over the same rows, in the
//!    same direction: a fill that encloses the view still covers it.
//! 4. What lies in the view is flattened with the device tolerance, a cubic in blocks
//!    of at most `MAX_QUADS` quadratics and about `BLOCK_LINES` lines.
//!
//! The work is that of the visible pieces, a few dozen bisection steps per cut, and
//! constant memory, whatever the coordinates. This follows the design of Skia's
//! `SkEdgeClipper` (monotone pieces, boundary lines for winding), without its
//! replacement of large curves by lines.
//!
//! Accuracy is that of `f64` evaluation of the device control points: a cut point is
//! within `ROOT_TOL` of the curve unless the curve moves farther between two adjacent
//! `f64` parameters (coordinates around 2^52 times the view and more); then the curve
//! between them is drawn as the line joining them, clipped like any line.

use crate::flatten::TOL;
use crate::flatten_simd::{Callback, FlattenCtx, LinePathEl, flatten_block, flatten_quad};
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{CubicBez, ParamCurve, ParamCurveExtrema, PathSeg, Point};
use fearless_simd::Simd;

/// How far, in device pixels, a cut point may be from where the curve crosses the edge.
const ROOT_TOL: f64 = 1.0 / 4096.0;

/// Lines a block of a cubic in the view flattens to, about at most: blocks bound the
/// flattener's buffer, not the number of lines.
const BLOCK_LINES: f64 = 256.0;

/// Flattens the device-space segment `seg` of a fill clipped to `rect` (left, top,
/// right, bottom), leaving the current point at its end.
#[cold]
#[inline(never)]
pub(crate) fn flatten_seg<S: Simd>(
    simd: S,
    seg: PathSeg,
    rect: [f64; 4],
    callback: &mut impl Callback,
    ctx: &mut FlattenCtx,
) {
    let mut clip = Clip {
        simd,
        rect,
        out: Out { callback, at: None },
        ctx,
        seg,
    };
    let mut ts = seg.extrema();
    ts.sort_by(f64::total_cmp);
    let (mut t0, mut p0) = (0.0, seg.start());
    for t in ts.into_iter().chain([1.0]) {
        let p1 = if t == 1.0 { seg.end() } else { seg.eval(t) };
        clip.piece(t0, p0, t, p1);
        (t0, p0) = (t, p1);
    }
    clip.out.move_to(seg.end());
}

/// Draws the device-space line `p0 p1` of a fill clipped to `rect`, leaving the
/// current point at `p1`.
#[cold]
#[inline(never)]
pub(crate) fn line(p0: Point, p1: Point, rect: [f64; 4], callback: &mut impl Callback) {
    let mut out = Out { callback, at: None };
    clip_line(&mut out, rect, p0, p1);
    out.move_to(p1);
}

/// Lines out, with a move only where one does not continue the last.
struct Out<'a, C> {
    callback: &'a mut C,
    at: Option<Point>,
}

impl<C: Callback> Out<'_, C> {
    fn move_to(&mut self, p: Point) {
        if self.at != Some(p) {
            self.callback.callback(LinePathEl::MoveTo(p));
            self.at = Some(p);
        }
    }

    fn line_to(&mut self, p: Point) {
        self.callback.callback(LinePathEl::LineTo(p));
        self.at = Some(p);
    }
}

/// Where a piece crosses an edge: a parameter and the point there, exactly on the
/// edge (`t0 == t1`); or, where the crossing lies between two adjacent `f64`
/// parameters, those two and the points there.
#[derive(Clone, Copy)]
struct Cut {
    t0: f64,
    p0: Point,
    t1: f64,
    p1: Point,
}

/// What a part of a piece between cuts is, by where it lies.
enum Region {
    /// Above, below or right of the view: dropped.
    Out,
    /// Left of the view: its rows' winding, on the left edge.
    Left,
    In,
}

fn region(rect: [f64; 4], a: Point, b: Point) -> Region {
    let [left, top, right, bottom] = rect;
    let m = a.midpoint(b);
    if m.y < top || m.y > bottom || m.x > right {
        Region::Out
    } else if m.x < left {
        Region::Left
    } else {
        Region::In
    }
}

/// `p` with coordinate `axis` (0: x, 1: y) set to `v`.
fn with(p: Point, axis: usize, v: f64) -> Point {
    if axis == 0 {
        Point::new(v, p.y)
    } else {
        Point::new(p.x, v)
    }
}

fn coord(p: Point, axis: usize) -> f64 {
    if axis == 0 { p.x } else { p.y }
}

/// The edges a piece from `a` to `b` may cross: axis and value.
fn edges(rect: [f64; 4]) -> [(usize, f64); 4] {
    let [left, top, right, bottom] = rect;
    [(1, top), (1, bottom), (0, left), (0, right)]
}

/// Whether a piece from `a` to `b` (monotone) crosses `v` on `axis` strictly inside.
fn crosses(a: Point, b: Point, axis: usize, v: f64) -> bool {
    let (a, b) = (coord(a, axis), coord(b, axis));
    (a < v && v < b) || (b < v && v < a)
}

/// Whether nothing of a piece from `a` to `b` (monotone) can reach the view.
fn misses(rect: [f64; 4], a: Point, b: Point) -> bool {
    let [_, top, right, bottom] = rect;
    a.y.max(b.y) <= top || a.y.min(b.y) >= bottom || a.x.min(b.x) >= right
}

/// Draws the line `a b` clipped to `rect`.
fn clip_line(out: &mut Out<'_, impl Callback>, rect: [f64; 4], a: Point, b: Point) {
    if misses(rect, a, b) || a == b {
        return;
    }
    let mut cuts = [a; 6];
    let mut n = 1;
    for (axis, v) in edges(rect) {
        if crosses(a, b, axis, v) {
            cuts[n] = at_coord(a, b, axis, v);
            n += 1;
        }
    }
    let cuts = &mut cuts[1..n];
    // Along a line, y and x change monotonically; ordered by projection, cuts far from
    // `a` would tie (100 and 0 against 1e30).
    let (sx, sy) = ((b.x - a.x).signum(), (b.y - a.y).signum());
    cuts.sort_by(|p, q| {
        (sy * p.y)
            .total_cmp(&(sy * q.y))
            .then((sx * p.x).total_cmp(&(sx * q.x)))
    });
    let mut from = a;
    for to in cuts.iter().copied().chain([b]) {
        line_part(out, rect, from, to);
        from = to;
    }
}

/// The point of `a b` on the boundary. Keep the endpoint distances and their
/// products as expansions: rounding a parameter near 0.5 loses an entire visible
/// crossing when the endpoints are huge, even if interpolation uses an FMA.
fn at_coord(a: Point, b: Point, axis: usize, v: f64) -> Point {
    let (ca, cb) = (coord(a, axis), coord(b, axis));
    // A power of two avoids rounding normal endpoint significands. Normalize
    // before subtracting opposite-sign endpoints, which could otherwise overflow.
    let exponent = ((ca.abs().max(cb.abs()).to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let shift = -exponent.max(-1022) - 2;
    let scale = if shift >= -1022 {
        f64::from_bits(((shift + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (shift + 1074))
    };
    let (ca, cb, v_scaled) = (ca * scale, cb * scale, v * scale);
    let (da, da_tail) = two_sum(v_scaled, -ca);
    let (db, db_tail) = two_sum(cb, -v_scaled);
    let (oa, ob) = (coord(a, 1 - axis), coord(b, 1 - axis));
    // Halving the other axis bounds products and partial sums even at f64::MAX.
    // The exact numerator is (v-ca)*ob + (cb-v)*oa. Its large terms must cancel
    // before rounding away the small contribution from the boundary coordinate.
    let mut expansion = [0.0; 8];
    let mut len = 0;
    for (distance, other) in [(da, ob), (da_tail, ob), (db, oa), (db_tail, oa)] {
        let other = other * 0.5;
        let product = distance * other;
        for term in [distance.mul_add(other, -product), product] {
            let mut sum = term;
            let mut next_len = 0;
            for i in 0..len {
                let (next, tail) = two_sum(sum, expansion[i]);
                if tail != 0.0 {
                    expansion[next_len] = tail;
                    next_len += 1;
                }
                sum = next;
            }
            expansion[next_len] = sum;
            len = next_len + 1;
        }
    }
    let other = (expansion[..len].iter().sum::<f64>() / (cb - ca)) * 2.0;
    // A boundary between the endpoints is a convex combination. Roundoff at
    // f64::MAX must not turn that finite intersection into infinity.
    with(
        with(a, 1 - axis, other.clamp(oa.min(ob), oa.max(ob))),
        axis,
        v,
    )
}

/// Error-free addition, apart from underflow of the roundoff term.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    (sum, (a - (sum - b_virtual)) + (b - b_virtual))
}

/// Draws a part of a line between cuts.
fn line_part(out: &mut Out<'_, impl Callback>, rect: [f64; 4], a: Point, b: Point) {
    match region(rect, a, b) {
        Region::Out => {}
        Region::Left => left_edge(out, rect, a, b),
        Region::In => {
            out.move_to(a);
            out.line_to(b);
        }
    }
}

/// The winding of a part from `a` to `b` left of the view, on its left edge.
fn left_edge(out: &mut Out<'_, impl Callback>, rect: [f64; 4], a: Point, b: Point) {
    if a.y != b.y {
        out.move_to(Point::new(rect[0], a.y));
        out.line_to(Point::new(rect[0], b.y));
    }
}

struct Clip<'a, S, C> {
    simd: S,
    rect: [f64; 4],
    out: Out<'a, C>,
    ctx: &'a mut FlattenCtx,
    seg: PathSeg,
}

impl<S: Simd, C: Callback> Clip<'_, S, C> {
    /// Draws the piece of the segment over `t0..t1`, monotone in x and y, from `p0`
    /// to `p1`.
    fn piece(&mut self, t0: f64, p0: Point, t1: f64, p1: Point) {
        if misses(self.rect, p0, p1) || p0 == p1 {
            return;
        }
        if let PathSeg::Line(_) = self.seg {
            clip_line(&mut self.out, self.rect, p0, p1);
            return;
        }
        // The piece's ends and cuts, each with whether the curve up to the next one
        // is drawn as the line joining them (a cut between adjacent parameters).
        let mut breaks = [(t0, p0, false); 10];
        let mut n = 1;
        for (axis, v) in edges(self.rect) {
            if crosses(p0, p1, axis, v) {
                let cut = self.cut(axis, v, t0, p0, t1, p1);
                breaks[n] = (cut.t0, cut.p0, cut.t1 != cut.t0);
                n += 1;
                if cut.t1 != cut.t0 {
                    breaks[n] = (cut.t1, cut.p1, false);
                    n += 1;
                }
            }
        }
        breaks[n] = (t1, p1, false);
        n += 1;
        let breaks = &mut breaks[..n];
        // An exact cut before a chord from the same parameter, so the chord is drawn.
        breaks.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
        for i in 0..n - 1 {
            let ((ta, pa, chord), (tb, pb, _)) = (breaks[i], breaks[i + 1]);
            if chord {
                clip_line(&mut self.out, self.rect, pa, pb);
            } else if tb > ta {
                self.part(ta, pa, tb, pb);
            }
        }
    }

    /// Draws a part of a piece between cuts, over `t0..t1` from `p0` to `p1`.
    fn part(&mut self, t0: f64, p0: Point, t1: f64, p1: Point) {
        // Separate boundary cuts can also leave adjacent parameters. Reconstructing
        // their controls from endpoint derivatives amplifies cancellation into an
        // enormous fictitious loop; the precision-floor rule applies to every span.
        let mid = t0 + 0.5 * (t1 - t0);
        if !(mid > t0 && mid < t1) {
            clip_line(&mut self.out, self.rect, p0, p1);
            return;
        }
        match region(self.rect, p0, p1) {
            Region::Out => {}
            Region::Left => left_edge(&mut self.out, self.rect, p0, p1),
            Region::In => {
                self.out.move_to(p0);
                match self.seg.subsegment(t0..t1) {
                    PathSeg::Line(_) => self.out.line_to(p1),
                    PathSeg::Quad(q) => {
                        flatten_quad(p0, q.p1, p1, self.rect, self.out.callback);
                    }
                    PathSeg::Cubic(c) => self.cubic(CubicBez::new(p0, c.p1, c.p2, p1)),
                }
                self.out.at = Some(p1);
            }
        }
    }

    /// Where the piece over `t0..t1` from `p0` to `p1` crosses `v` on `axis`, which
    /// it does strictly inside.
    fn cut(&self, axis: usize, v: f64, t0: f64, p0: Point, t1: f64, p1: Point) -> Cut {
        let below = coord(p0, axis) < v;
        let (mut lo, mut plo, mut hi, mut phi) = (t0, p0, t1, p1);
        loop {
            // The piece is monotone: between `lo` and `hi` it stays in their box.
            if (phi.x - plo.x).abs() <= ROOT_TOL && (phi.y - plo.y).abs() <= ROOT_TOL {
                let (a, b) = (coord(plo, axis), coord(phi, axis));
                let s = if a == b {
                    0.5
                } else {
                    ((v - a) / (b - a)).clamp(0.0, 1.0)
                };
                let t = lo + s * (hi - lo);
                let p = with(plo.lerp(phi, s), axis, v);
                return Cut {
                    t0: t,
                    p0: p,
                    t1: t,
                    p1: p,
                };
            }
            let mid = lo + 0.5 * (hi - lo);
            if !(mid > lo && mid < hi) {
                return Cut {
                    t0: lo,
                    p0: plo,
                    t1: hi,
                    p1: phi,
                };
            }
            let pm = self.seg.eval(mid);
            let c = coord(pm, axis);
            if c == v {
                return Cut {
                    t0: mid,
                    p0: pm,
                    t1: mid,
                    p1: pm,
                };
            }
            if (c < v) == below {
                (lo, plo) = (mid, pm);
            } else {
                (hi, phi) = (mid, pm);
            }
        }
    }

    /// Flattens the cubic `c` in the view in equal-parameter blocks, each within the
    /// flattener's `MAX_QUADS` quadratics and about `BLOCK_LINES` lines.
    fn cubic(&mut self, c: CubicBez) {
        // `MAX_QUADS` (16) quadratics approximate a cubic within the share of the
        // tolerance `estimate_num_quads` gives them while its third difference is at
        // most 16³ √432 times that share; over 1/k of the parameter it is k³ smaller.
        let third =
            (c.p1.to_vec2() * 3.0 - c.p0.to_vec2()) - (c.p2.to_vec2() * 3.0 - c.p3.to_vec2());
        let share = f64::from((TOL as f32) * 0.1);
        let quads = (third.hypot() / (432_f64.sqrt() * share)).cbrt() / 16.0;
        // A chord over 1/n of the parameter is within max |c''| / (8 n²) of a curve;
        // over 1/k of it, max |c''| is k² smaller.
        let second =
            |a: Point, b: Point, c: Point| (a.to_vec2() - 2.0 * b.to_vec2() + c.to_vec2()).hypot();
        let max_second = 6.0 * second(c.p0, c.p1, c.p2).max(second(c.p1, c.p2, c.p3));
        let lines = (max_second / (8.0 * TOL)).sqrt() / BLOCK_LINES;
        // In the view the cubic is monotone, so its control points lie within twice
        // its box's size of it and `k` is small; NaN gives one block.
        let k = quads.max(lines).ceil().max(1.0);
        let mut p0 = c.p0;
        let mut i = 1.0;
        while i <= k {
            let mut block = c.subsegment((i - 1.0) / k..i / k);
            block.p0 = p0;
            if i == k {
                block.p3 = c.p3;
            }
            flatten_block(self.simd, block, self.out.callback, self.ctx);
            p0 = block.p3;
            i += 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::{Level, dispatch};

    /// Check work as it is emitted, so a regression fails before retaining a large
    /// polyline or walking quadrillions of incorrectly reconstructed cubic blocks.
    struct CheckedOutput {
        at: Point,
        lines: usize,
    }

    impl Callback for CheckedOutput {
        fn callback(&mut self, el: LinePathEl) {
            match el {
                LinePathEl::MoveTo(p) => self.at = p,
                LinePathEl::LineTo(p) => {
                    self.lines += 1;
                    assert!(self.lines <= 1024, "unbounded clipped output");
                    for p in [self.at, p] {
                        assert!(p.x.is_finite() && p.y.is_finite(), "{p:?}");
                        assert!(
                            (-TOL..=100.0 + TOL).contains(&p.x)
                                && (-TOL..=100.0 + TOL).contains(&p.y),
                            "retained point outside the view: {p:?}"
                        );
                    }
                    self.at = p;
                }
            }
        }
    }

    fn checked(c: CubicBez) -> usize {
        let mut out = CheckedOutput { at: c.p0, lines: 0 };
        let mut ctx = FlattenCtx::default();
        dispatch!(Level::new(), simd => flatten_seg(
            simd, PathSeg::Cubic(c), [0.0, 0.0, 100.0, 100.0], &mut out, &mut ctx
        ));
        out.lines
    }

    #[test]
    fn line_intersections_keep_boundary_precision_across_scales() {
        for magnitude in [1e3, 1e20, 1e100, 1e200, 1e300, f64::MAX * 0.25] {
            for slope in [-2.0, -0.5, 0.5, 1.0, 2.0] {
                let a = Point::new(-magnitude, -magnitude * slope);
                let b = Point::new(2.0 * magnitude, 2.0 * magnitude * slope);
                for (a, b) in [(a, b), (b, a)] {
                    // Signed boundary coordinates also exercise translated views.
                    for v in [-100.0, -1.0, 0.0, 1.0, 100.0] {
                        for axis in [0, 1] {
                            let expected = if axis == 0 { v * slope } else { v / slope };
                            let point = at_coord(a, b, axis, v);
                            assert_eq!(coord(point, axis), v);
                            assert!(
                                (coord(point, 1 - axis) - expected).abs() <= 1e-12,
                                "{a:?} {b:?}, axis {axis}, boundary {v}: {point:?}, expected {expected}"
                            );
                        }
                    }
                }
            }
        }
        let m = f64::MAX;
        assert_eq!(
            at_coord(Point::new(-m, -m), Point::new(m, m), 0, 100.0),
            Point::new(100.0, 100.0)
        );
    }

    #[test]
    fn line_intersection_retains_a_cancelled_determinant() {
        // Integer endpoints: the determinant is exactly 1 although its two
        // products are about 2^104. This oracle comes from integer algebra.
        let m = (1_u64 << 52) as f64;
        for y_scale in [1.0, 1.0 / m, 2.0_f64.powi(800)] {
            let a = Point::new(m, (m - 1.0) * y_scale);
            let b = Point::new(1.0 - m, (2.0 - m) * y_scale);
            let expected = y_scale / (2.0 * m - 1.0);
            for (a, b) in [(a, b), (b, a)] {
                let actual = at_coord(a, b, 0, 0.0).y;
                assert!(
                    (actual - expected).abs() <= expected * 1e-14,
                    "{a:?} {b:?}: expected {expected}, actual {actual}"
                );
            }
        }
    }

    #[test]
    fn adjacent_boundary_cuts_do_not_reconstruct_a_huge_loop() {
        // Two boundary cuts leave adjacent parameters near 1.56e-105. Their
        // endpoint values cancel to the same visible point; derivative-based
        // subsegment reconstruction instead creates controls around +/-3e34.
        let c = CubicBez::new(
            (94.89934485887838, 12.046420206253082),
            (9.934373707850248e92, 1.5597482089393162e155),
            (-8.423477738983641e32, -1e260),
            (85.3758328038139, -8.604126786062645),
        );
        assert!(checked(c) < 64, "single clipped cubic work");
    }

    #[test]
    fn finite_controls_near_f64_limit_keep_visible_edges() {
        assert!(
            checked(CubicBez::new(
                (10.0, 10.0),
                (1e308, 1e308),
                (-1e308, 1e308),
                (90.0, 20.0)
            )) > 0,
            "finite near-limit controls must not erase visible edges"
        );
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }

        fn coord(&mut self, huge: bool) -> f64 {
            let sign = if self.next() < 0.5 { -1.0 } else { 1.0 };
            if huge {
                let exponent = (self.next() * 300.0).floor();
                let mantissa = if self.next() < 0.3 {
                    1.0
                } else {
                    self.next() * 9.0 + 1.0
                };
                sign * mantissa * 10_f64.powf(exponent)
            } else {
                -20.0 + self.next() * 140.0
            }
        }

        fn point(&mut self, huge: bool) -> Point {
            Point::new(self.coord(huge), self.coord(huge))
        }
    }

    #[test]
    fn deterministic_huge_cubics_keep_work_and_output_in_the_view() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        // Fixed seed/case count, including independent and cancelling controls.
        // The numeric exponent varies independently per coordinate, through 1e300.
        for case in 0..4096 {
            let p0 = rng.point(false);
            let mut controls = [Point::ZERO; 3];
            for p in &mut controls {
                let huge = rng.next() < 0.8;
                *p = rng.point(huge);
            }
            if case % 3 == 0 {
                controls[1] = Point::new(-controls[0].x, -controls[0].y);
            }
            let p3 = if case % 2 == 0 {
                rng.point(false)
            } else {
                controls[2]
            };
            checked(CubicBez::new(p0, controls[0], controls[1], p3));
        }
    }
}
