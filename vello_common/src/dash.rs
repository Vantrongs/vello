// Copyright 2023 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dashing as in `kurbo::dash` (kurbo 0.13.1 `stroke.rs`, `DashIterator`), copied so a
//! piece of a huge segment outside the view (flagged by `cull::SplitHuge`) costs a
//! constant: its whole dash periods are skipped in one step and its dashes are laid on
//! its chord. Kurbo walks every dash, so a dashed segment with huge coordinates took
//! time in proportion to its length. Every other segment is dashed exactly as kurbo
//! does.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{CubicBez, Line, ParamCurve, ParamCurveArclen, PathEl, PathSeg, Point, QuadBez};
use alloc::vec::Vec;

const DASH_ACCURACY: f64 = 1e-6;

/// Dashes `inner` like `kurbo::dash`, skipping the dash periods of the segments
/// flagged `true`.
pub(crate) fn dash<'a, T: Iterator<Item = (PathEl, bool)>>(
    inner: T,
    dash_offset: f64,
    dashes: &'a [f64],
) -> DashIterator<'a, T> {
    // Ensure that offset is positive and minimal by normalization using period
    let period = dashes.iter().sum();
    // The SVG spec requires odd-length dash arrays to be doubled to become even-length:
    // <https://www.w3.org/TR/SVG2/painting.html#StrokeDasharrayProperty>
    // This prevents gaps and dashes from swapping with one another as the offset increases.
    let period = if dashes.len() % 2 == 1 {
        2.0 * period
    } else {
        period
    };
    let dash_offset = dash_offset.rem_euclid(period);

    let mut dash_ix = 0;
    let mut dash_remaining = dashes[dash_ix] - dash_offset;
    let mut is_active = true;
    // Find place in dashes array for initial offset.
    while dash_remaining < 0.0 {
        dash_ix = (dash_ix + 1) % dashes.len();
        dash_remaining += dashes[dash_ix];
        is_active = !is_active;
    }
    DashIterator {
        inner,
        input_done: false,
        closepath_pending: false,
        dashes,
        dash_ix,
        init_dash_ix: dash_ix,
        init_dash_remaining: dash_remaining,
        init_is_active: is_active,
        is_active,
        state: DashState::NeedInput,
        current_seg: PathSeg::Line(Line::new(Point::ORIGIN, Point::ORIGIN)),
        t: 0.0,
        dash_remaining,
        seg_remaining: 0.0,
        start_pt: Point::ORIGIN,
        last_pt: Point::ORIGIN,
        stash: Vec::new(),
        stash_ix: 0,
        needs_moveto: true,
        period,
        skipping: false,
    }
}

/// An implementation of dashing as an iterator-to-iterator transformation.
pub(crate) struct DashIterator<'a, T> {
    inner: T,
    input_done: bool,
    closepath_pending: bool,
    dashes: &'a [f64],
    dash_ix: usize,
    init_dash_ix: usize,
    init_dash_remaining: f64,
    init_is_active: bool,
    is_active: bool,
    state: DashState,
    current_seg: PathSeg,
    t: f64,
    dash_remaining: f64,
    seg_remaining: f64,
    start_pt: Point,
    last_pt: Point,
    stash: Vec<PathEl>,
    stash_ix: usize,
    needs_moveto: bool,
    /// The length after which the dash state repeats.
    period: f64,
    /// Whether `current_seg` is the chord of a skipped segment (`set_segment`).
    skipping: bool,
}

#[derive(PartialEq, Eq)]
enum DashState {
    NeedInput,
    ToStash,
    Working,
    FromStash,
}

impl<T: Iterator<Item = (PathEl, bool)>> Iterator for DashIterator<'_, T> {
    type Item = PathEl;

    fn next(&mut self) -> Option<PathEl> {
        loop {
            match self.state {
                DashState::NeedInput => {
                    if self.input_done {
                        return None;
                    }
                    self.get_input();
                    if self.input_done {
                        return None;
                    }
                    self.state = DashState::ToStash;
                }
                DashState::ToStash => {
                    if let Some(el) = self.step() {
                        self.stash.push(el);
                    }
                }
                DashState::Working => {
                    if let Some(el) = self.step() {
                        return Some(el);
                    }
                }
                DashState::FromStash => {
                    if let Some(el) = self.stash.get(self.stash_ix) {
                        self.stash_ix += 1;
                        return Some(*el);
                    } else {
                        self.stash.clear();
                        self.stash_ix = 0;
                        if self.input_done {
                            return None;
                        }
                        if self.closepath_pending {
                            self.closepath_pending = false;
                            self.state = DashState::NeedInput;
                        } else {
                            self.state = DashState::ToStash;
                        }
                    }
                }
            }
        }
    }
}

impl<T: Iterator<Item = (PathEl, bool)>> DashIterator<'_, T> {
    fn get_input(&mut self) {
        loop {
            if self.closepath_pending {
                self.handle_closepath();
                break;
            }
            let Some((next_el, skip)) = self.inner.next() else {
                self.input_done = true;
                self.state = DashState::FromStash;
                return;
            };
            let p0 = self.last_pt;
            match next_el {
                PathEl::MoveTo(p) => {
                    if !self.stash.is_empty() {
                        self.state = DashState::FromStash;
                    }
                    self.start_pt = p;
                    self.last_pt = p;
                    self.reset_phase();
                    continue;
                }
                PathEl::LineTo(p1) => {
                    self.set_segment(PathSeg::Line(Line::new(p0, p1)), skip);
                    self.last_pt = p1;
                }
                PathEl::QuadTo(p1, p2) => {
                    self.set_segment(PathSeg::Quad(QuadBez::new(p0, p1, p2)), skip);
                    self.last_pt = p2;
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    self.set_segment(PathSeg::Cubic(CubicBez::new(p0, p1, p2, p3)), skip);
                    self.last_pt = p3;
                }
                PathEl::ClosePath => {
                    self.closepath_pending = true;
                    if p0 != self.start_pt {
                        self.set_segment(PathSeg::Line(Line::new(p0, self.start_pt)), false);
                        self.last_pt = self.start_pt;
                    } else {
                        self.handle_closepath();
                    }
                }
            }
            break;
        }
        self.t = 0.0;
    }

    /// Makes `seg` the current segment. One to `skip` (outside the view) is replaced
    /// by its chord with whole dash periods taken off its length: the dash state after
    /// it is the same, and its dashes, wherever they fall on the chord, paint nothing
    /// in view.
    fn set_segment(&mut self, seg: PathSeg, skip: bool) {
        self.skipping = skip;
        if !skip {
            self.seg_remaining = seg.arclen(DASH_ACCURACY);
            self.current_seg = seg;
            return;
        }
        // Relative accuracy: kurbo's absolute 1e-6 on a huge segment recurses to its
        // depth limit for nothing.
        let len = seg.arclen(DASH_ACCURACY.max(1e-9 * control_length(&seg)));
        let beyond = len - self.dash_remaining;
        self.seg_remaining = if self.period > 0.0 && beyond > self.period {
            self.dash_remaining + beyond % self.period
        } else {
            len
        };
        self.current_seg = PathSeg::Line(Line::new(seg.start(), seg.end()));
    }

    /// Move arc length forward to next event.
    fn step(&mut self) -> Option<PathEl> {
        let mut result = None;
        if self.state == DashState::ToStash && self.needs_moveto {
            self.needs_moveto = false;
            if self.is_active {
                result = Some(PathEl::MoveTo(self.current_seg.start()));
            } else {
                self.state = DashState::Working;
            }
        } else if self.dash_remaining < self.seg_remaining {
            // next transition is a dash transition
            let seg = self.current_seg.subsegment(self.t..1.0);
            let t1 = if self.skipping {
                self.dash_remaining / self.seg_remaining
            } else {
                seg.inv_arclen(self.dash_remaining, DASH_ACCURACY)
            };
            if self.is_active {
                let subseg = seg.subsegment(0.0..t1);
                result = Some(crate::cull::seg_to_el(&subseg));
                self.state = DashState::Working;
            } else {
                let p = seg.eval(t1);
                result = Some(PathEl::MoveTo(p));
            }
            self.is_active = !self.is_active;
            self.t += t1 * (1.0 - self.t);
            self.seg_remaining -= self.dash_remaining;
            self.dash_ix += 1;
            if self.dash_ix == self.dashes.len() {
                self.dash_ix = 0;
            }
            self.dash_remaining = self.dashes[self.dash_ix];
        } else {
            if self.is_active {
                let seg = self.current_seg.subsegment(self.t..1.0);
                result = Some(crate::cull::seg_to_el(&seg));
            }
            self.dash_remaining -= self.seg_remaining;
            self.get_input();
        }
        result
    }

    fn handle_closepath(&mut self) {
        if self.state == DashState::ToStash {
            // Have looped back without breaking a dash, just play it back
            self.stash.push(PathEl::ClosePath);
        } else if self.is_active {
            // connect with path in stash, skip MoveTo.
            self.stash_ix = 1;
        }
        self.state = DashState::FromStash;
        self.reset_phase();
    }

    fn reset_phase(&mut self) {
        self.dash_ix = self.init_dash_ix;
        self.dash_remaining = self.init_dash_remaining;
        self.is_active = self.init_is_active;
        self.needs_moveto = true;
    }
}

/// The length of the control polygon of `seg`, at least its arc length.
fn control_length(seg: &PathSeg) -> f64 {
    match *seg {
        PathSeg::Line(l) => l.p0.distance(l.p1),
        PathSeg::Quad(q) => q.p0.distance(q.p1) + q.p1.distance(q.p2),
        PathSeg::Cubic(c) => c.p0.distance(c.p1) + c.p1.distance(c.p2) + c.p2.distance(c.p3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cull::{Cull, SplitHuge};
    use crate::kurbo::{Affine, BezPath, ParamCurveNearest};

    const VIEW: [f64; 4] = [0.0, 0.0, 100.0, 100.0];

    fn mine(path: &BezPath, offset: f64, dashes: &[f64], rect: [f64; 4]) -> Vec<PathEl> {
        let split = SplitHuge::new(path.iter(), Cull::new(Affine::IDENTITY, rect), true);
        dash(split, offset, dashes).collect()
    }

    fn kurbo(path: &BezPath, offset: f64, dashes: &[f64]) -> Vec<PathEl> {
        crate::kurbo::dash(path.iter(), offset, dashes).collect()
    }

    #[test]
    fn dashes_as_kurbo_does_when_nothing_is_huge() {
        let mut p = BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((80.0, 10.0));
        p.quad_to((90.0, 50.0), (80.0, 90.0));
        p.curve_to((50.0, 120.0), (20.0, 60.0), (10.0, 90.0));
        p.move_to((200.0, 200.0));
        p.line_to((260.0, 200.0));
        p.curve_to((300.0, 240.0), (230.0, 280.0), (210.0, 230.0));
        p.close_path();
        // A segment after a close continues from the subpath's start.
        p.line_to((150.0, 150.0));
        p.move_to((-5000.0, -5000.0));
        p.line_to((-4000.0, -5000.0));
        p.line_to((-4000.0, -4000.0));
        p.close_path();
        for dashes in [&[5.0, 3.0][..], &[4.0], &[0.01, 2.0, 3.0]] {
            for offset in [0.0, 2.5, -1.0] {
                // A view that hides some segments, and one that holds them all.
                for rect in [[0.0, 0.0, 20.0, 20.0], [-1e6, -1e6, 1e6, 1e6]] {
                    assert_eq!(
                        mine(&p, offset, dashes, rect),
                        kurbo(&p, offset, dashes),
                        "{dashes:?} {offset} {rect:?}"
                    );
                }
            }
        }
    }

    /// The dashes in `out` (lines only) as line pieces near the view, and their ends.
    fn dashes_near_view(out: &[PathEl]) -> (Vec<Line>, Vec<Point>) {
        let near = |p: Point| (-1.0..=101.0).contains(&p.x) && (-1.0..=101.0).contains(&p.y);
        let (mut lines, mut ends) = (Vec::new(), Vec::new());
        let mut last = Point::ORIGIN;
        for (i, el) in out.iter().enumerate() {
            match *el {
                PathEl::MoveTo(q) => {
                    ends.push(q);
                    if i > 0 {
                        ends.push(last);
                    }
                    last = q;
                }
                PathEl::LineTo(q) => {
                    let l = Line::new(last, q);
                    let b = [
                        l.p0.x.min(q.x),
                        l.p0.y.min(q.y),
                        l.p0.x.max(q.x),
                        l.p0.y.max(q.y),
                    ];
                    if b[2] >= -1.0 && b[3] >= -1.0 && b[0] <= 101.0 && b[1] <= 101.0 {
                        lines.push(l);
                    }
                    last = q;
                }
                _ => panic!("lines only: {el:?}"),
            }
        }
        ends.push(last);
        ends.retain(|&e| near(e));
        (lines, ends)
    }

    fn on_dash(lines: &[Line], p: Point) -> bool {
        lines.iter().any(|l| l.nearest(p, 1e-9).distance_sq < 1e-12)
    }

    /// Along `line`, inside the view, `a` and `b` paint the same dashes, up to `tol`
    /// at the dash ends.
    fn same_dashes_in_view(a: &[PathEl], b: &[PathEl], line: Line, tol: f64) {
        let ((la, ea), (lb, eb)) = (dashes_near_view(a), dashes_near_view(b));
        let mut checked = 0;
        for i in 0..=100_000 {
            let p = line.eval(f64::from(i) / 100_000.0);
            if !(0.0..=100.0).contains(&p.x) || !(0.0..=100.0).contains(&p.y) {
                continue;
            }
            if ea.iter().chain(&eb).any(|e| e.distance(p) < tol) {
                continue;
            }
            checked += 1;
            assert_eq!(on_dash(&la, p), on_dash(&lb, p), "at {p:?}");
        }
        assert!(checked > 100, "{checked} points checked");
    }

    /// Dashing a line `len` long with a period `period` gives at most this many
    /// elements (a move and a line per dash).
    fn dash_elements(len: f64, period: f64) -> usize {
        (2.0 * len / period) as usize + 2
    }

    #[test]
    fn a_skipped_segment_keeps_the_dash_phase() {
        let dashes = [7.0, 3.0];
        // A line 1e6 long outside the view, then one through it.
        let mut p = BezPath::new();
        p.move_to((-1e6 + 0.3, 50.0));
        p.line_to((-200.0, 50.0));
        p.line_to((100.0, 50.0));
        let (a, b) = (mine(&p, 1.5, &dashes, VIEW), kurbo(&p, 1.5, &dashes));
        assert!(
            a.len() < dash_elements(300.0, 10.0) + 4,
            "{} elements",
            a.len()
        );
        let visible = Line::new((-200.0, 50.0), (100.0, 50.0));
        same_dashes_in_view(&a, &b, visible, 1e-6);
        // A closed subpath whose first and closing lines, 1e6 long, cross the view:
        // only their pieces that meet the view, each at most as long as the view is
        // wide, are dashed.
        let mut p = BezPath::new();
        p.move_to((50.0, 50.0));
        p.line_to((1e6, 50.0));
        p.line_to((1e6, 60.0));
        p.close_path();
        let (a, b) = (mine(&p, 0.0, &dashes, VIEW), kurbo(&p, 0.0, &dashes));
        let bound = 2 * 4 * dash_elements(100.0, 10.0);
        assert!(a.len() < bound && bound < b.len(), "{} elements", a.len());
        same_dashes_in_view(&a, &b, Line::new((50.0, 50.0), (150.0, 50.0)), 1e-6);
        let closing = Line::new((1e6, 60.0), (50.0, 50.0));
        let in_view = Line::new(closing.eval((1e6 - 100.0) / (1e6 - 50.0)), closing.p1);
        same_dashes_in_view(&a, &b, in_view, 1e-6);
    }

    #[test]
    fn a_skipped_curve_keeps_the_dash_phase() {
        let dashes = [7.0, 3.0];
        let mut p = BezPath::new();
        p.move_to((-1e6, 50.0));
        p.curve_to((-6e5, -2e5), (-4e5, 3e5), (-300.0, 50.0));
        p.line_to((100.0, 50.0));
        let flat = |els: Vec<PathEl>| {
            let mut out = Vec::new();
            crate::kurbo::flatten(els, 0.01, |el| out.push(el));
            out
        };
        let (a, b) = (
            flat(mine(&p, 0.0, &dashes, VIEW)),
            flat(kurbo(&p, 0.0, &dashes)),
        );
        // The skipped cubic's length is measured to 1e-9 of its size, not to 1e-6.
        same_dashes_in_view(&a, &b, Line::new((-300.0, 50.0), (100.0, 50.0)), 1e-2);
    }

    #[test]
    fn a_huge_dashed_cubic_costs_little() {
        let mut p = BezPath::new();
        p.move_to((10.0, 10.0));
        p.curve_to((1e24, 10.0), (-1e24, 10.0), (20.0, 10.0));
        let out = mine(&p, 0.0, &[2.0, 2.0], VIEW);
        // Three passes through the view, plus a few elements for each of the pieces
        // skipped on the way down from 1e24 (about 75 halvings per pass).
        let bound = 3 * (4 * dash_elements(100.0, 4.0) + 6 * 80);
        assert!(out.len() < bound, "{} elements", out.len());
        // Its three passes through the view (along y = 10) are drawn.
        let mut flat = Vec::new();
        crate::kurbo::flatten(out, 1e-3, |el| flat.push(el));
        let (lines, _) = dashes_near_view(&flat);
        let on = (0..100).filter(|&x| on_dash(&lines, Point::new(f64::from(x) + 0.5, 10.0)));
        assert!(on.count() >= 40);
    }
}
