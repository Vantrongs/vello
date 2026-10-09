// Copyright 2023 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dashing as in `kurbo::dash` (kurbo 0.13.1 `stroke.rs`, `DashIterator`), copied so
//! its work stays bounded by the view:
//!
//! - A piece of a segment outside the view (flagged by `cull::SplitHuge`) is replaced
//!   by its chord, drawn solid: its whole dash periods are skipped in one step, and
//!   it paints nothing in view whatever its dashes. Kurbo walks every dash, so a
//!   dashed segment with huge coordinates took time in proportion to its length.
//! - A segment along which dashes and gaps (the pattern's entries walked) average
//!   under `1 / DENSE_ENTRIES_PER_PX` device pixels, or beyond the `PATH_ENTRIES` a
//!   path may walk, is drawn solid, its dash phase carried on as if dashed. Kurbo
//!   emits every dash, so a pattern far finer than a pixel (a tiny scale, a transform
//!   that compresses one direction, or many tiny entries) emitted dashes in
//!   proportion to the inverse of their device size. `MuPDF` draws such patterns
//!   solid too, below half a pixel per period at the transform's largest stretch
//!   (1.28 `draw-path.c`, `do_flatten_stroke`).
//!
//! Every other segment is dashed exactly as kurbo does.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{
    Affine, CubicBez, Line, ParamCurve, ParamCurveArclen, PathEl, PathSeg, Point, QuadBez,
};
use alloc::vec::Vec;

const DASH_ACCURACY: f64 = 1e-6;

/// Pattern entries (dashes and gaps) per device pixel, on average along a segment,
/// beyond which the segment is drawn solid: for a pattern of a dash and a gap, a
/// period under a sixteenth of a pixel. Such dashes are a fine texture no output
/// resolves, and drawing them all could take any amount of memory. Entries, not
/// periods, are counted: each is an element out, and a period can hold any number.
const DENSE_ENTRIES_PER_PX: f64 = 32.0;

/// Pattern entries one path may walk; segments beyond are drawn solid. Segments
/// larger than the view are split and their hidden pieces skipped (`cull::SplitHuge`),
/// so drawings stay far below this; it bounds the dashes of a path whose geometry is
/// numerically noisy (coordinates near the limits of `f64` under a nearly singular
/// transform), whose split pieces can each look long on the device. A dash and its
/// gap take about 64 bytes as a thin outline and 300 as an expanded one.
const PATH_ENTRIES: f64 = (1 << 22) as f64;

/// Dashes `inner` like `kurbo::dash`, skipping the dash periods of the segments
/// flagged `true`; `affine` maps the path to device space.
pub(crate) fn dash<'a, T: Iterator<Item = (PathEl, bool)>>(
    inner: T,
    dash_offset: f64,
    dashes: &'a [f64],
    affine: Affine,
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
    let entries = if dashes.len() % 2 == 1 {
        2 * dashes.len()
    } else {
        dashes.len()
    };

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
        entries_per_period: entries as f64,
        dense: None,
        affine,
        entries_left: PATH_ENTRIES,
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
    /// The pattern entries walked along `period`.
    entries_per_period: f64,
    /// Whether `current_seg` is drawn solid (`set_segment`), and if so how far it is
    /// out.
    dense: Option<Solid>,
    affine: Affine,
    /// Pattern entries the path may still walk (`PATH_ENTRIES`).
    entries_left: f64,
}

/// How far a segment drawn solid is out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Solid {
    Start,
    /// The move to its start, which a dash starting with it needs.
    Moved,
    /// The segment, held back with the subpath's first dash, which ended inside it;
    /// the dash running on past its end needs a move there.
    Stashed,
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
    /// by its chord, drawn solid, with whole dash periods taken off its length: the
    /// dash state after it is the same, and it paints nothing in view, as its control
    /// box misses the view widened by the outline's reach. One whose dashes are too
    /// dense (see the module) keeps its length less whole periods too, and is drawn
    /// solid.
    fn set_segment(&mut self, seg: PathSeg, skip: bool) {
        if skip {
            // Relative accuracy: kurbo's absolute 1e-6 on a huge segment recurses to
            // its depth limit for nothing.
            let len = seg.arclen(DASH_ACCURACY.max(1e-9 * control_length(&seg)));
            self.dense = Some(Solid::Start);
            self.seg_remaining = self.less_whole_periods(len);
            self.current_seg = PathSeg::Line(Line::new(seg.start(), seg.end()));
            return;
        }
        let len = seg.arclen(DASH_ACCURACY);
        self.current_seg = seg;
        let entries = len / self.period * self.entries_per_period;
        if self.period > 0.0
            && (entries > self.entries_left
                || entries > DENSE_ENTRIES_PER_PX * (self.device_length(&seg) + 1.0))
        {
            self.dense = Some(Solid::Start);
            self.seg_remaining = self.less_whole_periods(len);
        } else {
            self.dense = None;
            self.entries_left -= entries;
            self.seg_remaining = len;
        }
    }

    /// `len` less as many whole periods as leave it longer than the current dash, so
    /// walking it gives the dash state walking `len` would.
    fn less_whole_periods(&self, len: f64) -> f64 {
        let (dash, period) = (self.dash_remaining, self.period);
        if !(period > 0.0 && len - dash > period) {
            return len;
        }
        // `len - dash` would round away the small remainder of a huge length: reduce
        // `len` first, exactly (`%` is exact), then subtract.
        dash + (len % period - dash).rem_euclid(period)
    }

    /// An upper bound on the device length of `seg`: its control polygon's.
    fn device_length(&self, seg: &PathSeg) -> f64 {
        control_length(&(self.affine * *seg))
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
        } else if let Some(solid) = self.dense {
            if solid == Solid::Stashed {
                let end = self.current_seg.end();
                self.get_input();
                return Some(PathEl::MoveTo(end));
            }
            if solid == Solid::Start && !self.is_active {
                // A dash starts with the segment.
                self.dense = Some(Solid::Moved);
                return Some(PathEl::MoveTo(self.current_seg.start()));
            }
            result = Some(crate::cull::seg_to_el(&self.current_seg));
            let stashing = self.state == DashState::ToStash;
            // The dash state at its end, as walking it would leave it.
            while self.dash_remaining < self.seg_remaining {
                if self.is_active {
                    self.state = DashState::Working;
                }
                self.is_active = !self.is_active;
                self.seg_remaining -= self.dash_remaining;
                self.dash_ix += 1;
                if self.dash_ix == self.dashes.len() {
                    self.dash_ix = 0;
                }
                self.dash_remaining = self.dashes[self.dash_ix];
            }
            self.dash_remaining -= self.seg_remaining;
            if stashing && self.state == DashState::Working && self.is_active {
                self.dense = Some(Solid::Stashed);
            } else {
                self.get_input();
            }
        } else if self.dash_remaining < self.seg_remaining {
            // next transition is a dash transition
            let seg = self.current_seg.subsegment(self.t..1.0);
            let t1 = seg.inv_arclen(self.dash_remaining, DASH_ACCURACY);
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
        starts_with_a_move(dash(split, offset, dashes, Affine::IDENTITY).collect())
    }

    /// A stroker starts a line out first from the origin.
    fn starts_with_a_move(out: Vec<PathEl>) -> Vec<PathEl> {
        assert!(
            matches!(out.first(), None | Some(PathEl::MoveTo(_))),
            "{out:?}"
        );
        out
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
        let flat = |els: Vec<PathEl>| {
            let mut out = Vec::new();
            crate::kurbo::flatten(els, 0.01, |el| out.push(el));
            out
        };
        for dashes in [&[5.0, 3.0][..], &[4.0], &[0.01, 2.0, 3.0]] {
            for offset in [0.0, 2.5, -1.0] {
                // A view that holds every segment.
                let all = [-1e6, -1e6, 1e6, 1e6];
                assert_eq!(
                    mine(&p, offset, dashes, all),
                    kurbo(&p, offset, dashes),
                    "{dashes:?} {offset}"
                );
                // A view that holds part of the first line: the segments beyond are
                // split and their hidden pieces skipped; the dashes in view are the
                // same, up to rounding at their ends.
                let (a, b) = (
                    flat(mine(&p, offset, dashes, [0.0, 0.0, 20.0, 20.0])),
                    flat(kurbo(&p, offset, dashes)),
                );
                same_dashes_in_view(&a, &b, Line::new((10.0, 10.0), (20.0, 10.0)), 1e-9);
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

    /// The review's case: a hidden line 2^54 - 200 long before one through the view.
    /// Its length is 4 modulo the period, so the visible line starts at phase 4;
    /// `len - dash` before the modulo rounded that to 3.
    #[test]
    fn a_huge_skipped_length_keeps_its_remainder() {
        let dashes = [3.0, 7.0];
        let far = -(2_f64.powi(54));
        let input = [
            (PathEl::MoveTo(Point::new(far, 50.0)), false),
            (PathEl::LineTo(Point::new(-200.0, 50.0)), true),
            (PathEl::LineTo(Point::new(100.0, 50.0)), false),
        ];
        let a =
            starts_with_a_move(dash(input.into_iter(), 0.0, &dashes, Affine::IDENTITY).collect());
        let mut short = BezPath::new();
        short.move_to((-200.0, 50.0));
        short.line_to((100.0, 50.0));
        let b = kurbo(&short, 4.0, &dashes);
        same_dashes_in_view(&a, &b, Line::new((-200.0, 50.0), (100.0, 50.0)), 1e-6);
        // The same through `SplitHuge`, which skips the line in pieces.
        let mut p = BezPath::new();
        p.move_to((far, 50.0));
        p.line_to((-200.0, 50.0));
        p.line_to((100.0, 50.0));
        let a = mine(&p, 0.0, &dashes, VIEW);
        same_dashes_in_view(&a, &b, Line::new((-200.0, 50.0), (100.0, 50.0)), 1e-6);
    }

    /// A segment whose dashes are far below a pixel on the device is drawn whole, and
    /// the next one starts at the phase walking it would leave.
    #[test]
    fn dense_dashes_are_drawn_solid_in_phase() {
        let dashes = [3.0, 7.0];
        // x is compressed 1000-fold: the first line, 1000.3 long in path space (100
        // periods), is 1 px on the device; the second, vertical, is not compressed.
        let affine = Affine::scale_non_uniform(1e-3, 1.0);
        let mut p = BezPath::new();
        p.move_to((-1000.3, 50.0));
        p.line_to((0.0, 50.0));
        p.line_to((0.0, 100.0));
        let wide = Cull::new(affine, [-1e6, -1e6, 1e6, 1e6]);
        let split = SplitHuge::new(p.iter(), wide, true);
        let a = starts_with_a_move(dash(split, 1.5, &dashes, affine).collect());
        // Starting inside a dash (offset 1.5 < 3), the first line is one stroke (out
        // last: kurbo holds a subpath's first dash back to join a closing one).
        let whole = [
            PathEl::MoveTo(Point::new(-1000.3, 50.0)),
            PathEl::LineTo(Point::new(0.0, 50.0)),
        ];
        assert!(a.windows(2).any(|w| w == whole), "{a:?}");
        // The vertical line is dashed from phase (1.5 + 1000.3) mod 10 = 1.8, as kurbo
        // dashes the whole path.
        let b = kurbo(&p, 1.5, &dashes);
        same_dashes_in_view(&a, &b, Line::new((0.0, 50.0), (0.0, 100.0)), 1e-6);
        // Starting in a gap (offset 4), the first line still is one stroke.
        let split = SplitHuge::new(p.iter(), wide, true);
        let a = starts_with_a_move(dash(split, 4.0, &dashes, affine).collect());
        assert!(a.windows(2).any(|w| w == whole), "{a:?}");
        let b = kurbo(&p, 4.0, &dashes);
        same_dashes_in_view(&a, &b, Line::new((0.0, 50.0), (0.0, 100.0)), 1e-6);
    }
}
