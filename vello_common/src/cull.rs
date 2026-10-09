// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Keeps the cost of flattening, stroking and dashing a segment bounded by the view
//! rather than by the size of its coordinates.
//!
//! Kurbo's `flatten` reserves memory for a cubic in proportion to the cube root of its
//! size and emits lines in proportion to the square root, and dashing walks every dash,
//! so one cubic with control points at 1e24 asks for gigabytes. Segments wider or
//! taller than [`SPLIT_EXTENT`] device pixels are therefore split (de Casteljau, at the
//! middle) until each piece is no larger than the view or its control box misses the
//! view. Real drawings stay below that size, so their segments pass through unchanged.

use crate::kurbo::{Affine, CubicBez, Line, ParamCurve, PathEl, PathSeg, Point, QuadBez};
use alloc::vec::Vec;

/// Segments whose device control box is wider or taller than this many pixels are
/// split. A cubic this size flattens to a few thousand lines at most.
pub(crate) const SPLIT_EXTENT: f64 = 131_072.0;

/// Halvings after which a piece that still crosses the view is drawn as its chord:
/// 2^128 pixels is beyond any `f32` coordinate under a sane transform.
const MAX_DEPTH: u32 = 128;

/// Splits of one segment after which its remaining pieces are drawn as chords.
const MAX_SPLITS: u32 = 4096;

/// The view in device pixels, and the transform from path to device space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cull {
    affine: Affine,
    /// Left, top, right, bottom: geometry beyond them paints nothing.
    rect: [f64; 4],
}

impl Cull {
    pub(crate) fn new(affine: Affine, rect: [f64; 4]) -> Self {
        Self { affine, rect }
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
}

fn exceeds(b: [f64; 4], extent: f64) -> bool {
    (b[2] - b[0]).max(b[3] - b[1]) > extent
}

/// A path whose segments larger than `SPLIT_EXTENT` are split into pieces that are
/// each no larger than the view or outside it. Each element comes with whether it is
/// such a piece outside the view.
///
/// A piece outside the view becomes its chord, unless `keep_hidden` (dashing needs its
/// arc length; [`crate::dash::dash`] then skips it). The chord paints the same pixels
/// inside the view: for a fill, the region between a curve and its chord lies in the
/// control hull; for a stroke, `Cull`'s rect is widened by the outline's reach, which
/// also covers the joins at the piece's ends, whose angle the chord changes. Lines are
/// split only for dashing; a fill or an undashed stroke handles them in constant time.
pub(crate) struct SplitHuge<I> {
    inner: I,
    cull: Cull,
    keep_hidden: bool,
    start: Point,
    last: Point,
    /// Pieces still to emit with their depth, the next one on top.
    stack: Vec<(PathSeg, u32)>,
    splits: u32,
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
            start: Point::ZERO,
            last: Point::ZERO,
            stack: Vec::new(),
            splits: 0,
            close_pending: false,
        }
    }

    /// The element for `seg` (a segment at depth 0, else a piece of one), or `None`
    /// once its halves are on the stack.
    fn piece(&mut self, seg: PathSeg, depth: u32) -> Option<(PathEl, bool)> {
        let b = self.cull.device_box(&seg);
        if depth == 0 && !exceeds(b, SPLIT_EXTENT) {
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
            return Some((seg_to_el(&seg), false));
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
                        || !exceeds(self.cull.device_box(&closing), SPLIT_EXTENT)
                    {
                        return Some((el, false));
                    }
                    // Dashing walks the closing line too: split it as an explicit line.
                    self.close_pending = true;
                    closing
                }
            };
            self.splits = 0;
            if let Some(el) = self.piece(seg, 0) {
                return Some(el);
            }
        }
    }
}

/// The path with huge segments split, for a fill or an undashed stroke.
pub(crate) fn split_huge<I: Iterator<Item = PathEl>>(
    path: impl IntoIterator<IntoIter = I>,
    cull: Cull,
) -> impl Iterator<Item = PathEl> {
    SplitHuge::new(path, cull, false).map(|(el, _)| el)
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
        // Dashing also splits the 1e9-long line, and nothing else.
        let out = split(&p, Affine::IDENTITY, true);
        same(&out[..8], &p.elements()[..8]);
        assert!(out.len() > p.elements().len());
        check_pieces(&out[8..], Point::ZERO, Point::new(1e9, 7.0), true);
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
}
