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

/// The accuracy dashing measures lengths along segments to.
pub const DASH_ACCURACY: f64 = 1e-6;

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

/// A dash pattern as dashing walks it along each subpath: its entries drawn and
/// skipped in turn, the first drawn, from the offset on. A pattern of an odd number of
/// entries is walked twice per period, the second time with what is drawn and what is
/// skipped swapped.
#[derive(Clone, Copy, Debug)]
pub struct Pattern<'a> {
    dashes: &'a [f64],
    /// The length after which the dash state repeats.
    period: f64,
    /// The pattern entries walked along `period`.
    entries_per_period: f64,
    /// Where each subpath starts.
    start: Phase,
}

/// Where dashing is in a [`Pattern`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase {
    /// The entry.
    ix: usize,
    /// The length left of it.
    remaining: f64,
    /// Whether it is drawn.
    active: bool,
}

impl<'a> Pattern<'a> {
    /// The pattern of `dashes` from `dash_offset` on, which must not be empty.
    ///
    /// A pattern with a negative entry walks back forever; one whose entries are not
    /// finite or sum to zero draws every segment whole.
    pub fn new(dash_offset: f64, dashes: &'a [f64]) -> Self {
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
        Self {
            dashes,
            period,
            entries_per_period: entries as f64,
            start: Phase {
                ix: dash_ix,
                remaining: dash_remaining,
                active: is_active,
            },
        }
    }

    /// Where each subpath starts.
    pub fn start(&self) -> Phase {
        self.start
    }

    /// Move `phase` to the start of the next entry.
    fn next(&self, phase: &mut Phase) {
        phase.active = !phase.active;
        phase.ix += 1;
        if phase.ix == self.dashes.len() {
            phase.ix = 0;
        }
        phase.remaining = self.dashes[phase.ix];
    }

    /// Walk `len` along from `phase`, leave in `len` what is walked of the last entry
    /// reached, and return whether a drawn entry ended on the way.
    fn walk(&self, phase: &mut Phase, len: &mut f64) -> bool {
        let mut ended = false;
        while phase.remaining < *len {
            ended |= phase.active;
            *len -= phase.remaining;
            self.next(phase);
        }
        phase.remaining -= *len;
        ended
    }

    /// `len` less as many whole periods as leave it longer than what is left of the
    /// entry at `phase`, so walking it gives the dash state walking `len` would.
    fn less_whole_periods(&self, phase: &Phase, len: f64) -> f64 {
        let (dash, period) = (phase.remaining, self.period);
        if !(period > 0.0 && len - dash > period) {
            return len;
        }
        // `len - dash` would round away the small remainder of a huge length: reduce
        // `len` first, exactly (`%` is exact), then subtract.
        dash + (len % period - dash).rem_euclid(period)
    }

    /// Walk `len` along from `phase` as dashing walks a segment it draws solid, in
    /// work that does not grow with `len`.
    pub fn skip(&self, phase: &mut Phase, len: f64) {
        self.walk(phase, &mut self.less_whole_periods(phase, len));
    }

    /// Return where the entry dashing from `phase` reaches the point `at` along in
    /// starts and ends, as lengths along from `phase`, and whether it is drawn; it
    /// starts before `phase` if `phase` is inside it.
    pub fn entry_around(&self, mut phase: Phase, at: f64) -> (f64, f64, bool) {
        let mut left = self.less_whole_periods(&phase, at);
        let mut walked = at - left;
        let mut start = walked - (self.dashes[phase.ix] - phase.remaining);
        while phase.remaining < left {
            walked += phase.remaining;
            left -= phase.remaining;
            start = walked;
            self.next(&mut phase);
        }
        (start, walked + phase.remaining, phase.active)
    }
}

/// What dashing does with a path element.
#[derive(Clone, Copy, Debug)]
pub enum Walk {
    /// Starts a subpath, and the pattern over.
    Move,
    /// Walks a segment.
    Segment(PathSeg),
    /// Closes the subpath, walking the line back to its start if it ends elsewhere; the
    /// pattern starts over after it, see `DashIterator::handle_closepath`.
    Close(Option<PathSeg>),
}

/// Where a path's subpath starts and its last segment ends, as dashing walks it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Subpaths {
    start: Point,
    last: Point,
}

impl Subpaths {
    /// Return the line back to where the subpath starts, unless it ends there.
    pub fn open(&self) -> Option<PathSeg> {
        (self.last != self.start).then(|| PathSeg::Line(Line::new(self.last, self.start)))
    }

    /// Walk `el`.
    pub fn walk(&mut self, el: PathEl) -> Walk {
        let p0 = self.last;
        match el {
            PathEl::MoveTo(p) => {
                self.start = p;
                self.last = p;
                Walk::Move
            }
            PathEl::LineTo(p1) => {
                self.last = p1;
                Walk::Segment(PathSeg::Line(Line::new(p0, p1)))
            }
            PathEl::QuadTo(p1, p2) => {
                self.last = p2;
                Walk::Segment(PathSeg::Quad(QuadBez::new(p0, p1, p2)))
            }
            PathEl::CurveTo(p1, p2, p3) => {
                self.last = p3;
                Walk::Segment(PathSeg::Cubic(CubicBez::new(p0, p1, p2, p3)))
            }
            PathEl::ClosePath => {
                let closing = self.open();
                self.last = self.start;
                Walk::Close(closing)
            }
        }
    }
}

/// Return the segments dashing with `pattern` walks along `path`, each with the
/// phase it starts at, in work that follows the number of segments: the phases are
/// those dashing reaches up to rounding, as it walks most segments dash by dash.
///
/// A closing read before any segment since the last closing, or since the path's
/// start, walks the last segment walked again from its start, see
/// `DashIterator::get_input`, where `step` takes it up from the state the closing
/// leaves; it is given whole, with the phase a subpath starts at, though dashing
/// walks it only so far and joins what it draws to what it drew last.
pub fn phases<'a>(
    path: impl IntoIterator<Item = PathEl> + 'a,
    pattern: &'a Pattern<'_>,
) -> impl Iterator<Item = (PathSeg, Phase)> + 'a {
    let mut subpaths = Subpaths::default();
    let mut phase = pattern.start();
    let mut last = PathSeg::Line(Line::new(Point::ORIGIN, Point::ORIGIN));
    // Whether a closing walks `last` again.
    let mut again = true;
    path.into_iter().filter_map(move |el| {
        let (seg, closes) = match subpaths.walk(el) {
            Walk::Move => {
                phase = pattern.start();
                return None;
            }
            Walk::Segment(seg) => (seg, false),
            Walk::Close(Some(seg)) => (seg, true),
            Walk::Close(None) => {
                phase = pattern.start();
                return core::mem::replace(&mut again, true).then_some((last, phase));
            }
        };
        let at = phase;
        pattern.skip(&mut phase, seg.arclen(DASH_ACCURACY));
        if closes {
            phase = pattern.start();
        }
        (last, again) = (seg, closes);
        Some((seg, at))
    })
}

/// Dashes `inner` like `kurbo::dash`, skipping the dash periods of the segments
/// flagged `true`; `affine` maps the path to device space.
pub(crate) fn dash<'a, T: Iterator<Item = (PathEl, bool)>>(
    inner: T,
    dash_offset: f64,
    dashes: &'a [f64],
    affine: Affine,
) -> DashIterator<'a, T> {
    let pattern = Pattern::new(dash_offset, dashes);
    DashIterator {
        inner,
        input_done: false,
        closepath_pending: false,
        pattern,
        phase: pattern.start(),
        state: DashState::NeedInput,
        current_seg: PathSeg::Line(Line::new(Point::ORIGIN, Point::ORIGIN)),
        t: 0.0,
        seg_remaining: 0.0,
        subpaths: Subpaths::default(),
        stash: Vec::new(),
        stash_ix: 0,
        needs_moveto: true,
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
    pattern: Pattern<'a>,
    phase: Phase,
    state: DashState,
    current_seg: PathSeg,
    t: f64,
    seg_remaining: f64,
    subpaths: Subpaths,
    stash: Vec<PathEl>,
    stash_ix: usize,
    needs_moveto: bool,
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
            match self.subpaths.walk(next_el) {
                Walk::Move => {
                    if !self.stash.is_empty() {
                        self.state = DashState::FromStash;
                    }
                    self.reset_phase();
                    continue;
                }
                Walk::Segment(seg) => self.set_segment(seg, skip),
                Walk::Close(closing) => {
                    self.closepath_pending = true;
                    match closing {
                        Some(closing) => self.set_segment(closing, false),
                        None => self.handle_closepath(),
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
            self.seg_remaining = self.pattern.less_whole_periods(&self.phase, len);
            self.current_seg = PathSeg::Line(Line::new(seg.start(), seg.end()));
            return;
        }
        let len = seg.arclen(DASH_ACCURACY);
        self.current_seg = seg;
        let (period, entries_per_period) = (self.pattern.period, self.pattern.entries_per_period);
        let entries = len / period * entries_per_period;
        if period > 0.0
            && (entries > self.entries_left
                || entries > DENSE_ENTRIES_PER_PX * (self.device_length(&seg) + 1.0))
        {
            self.dense = Some(Solid::Start);
            self.seg_remaining = self.pattern.less_whole_periods(&self.phase, len);
        } else {
            self.dense = None;
            self.entries_left -= entries;
            self.seg_remaining = len;
        }
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
            if self.phase.active {
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
            if solid == Solid::Start && !self.phase.active {
                // A dash starts with the segment.
                self.dense = Some(Solid::Moved);
                return Some(PathEl::MoveTo(self.current_seg.start()));
            }
            result = Some(crate::cull::seg_to_el(&self.current_seg));
            let stashing = self.state == DashState::ToStash;
            // The dash state at its end, as walking it would leave it.
            if self.pattern.walk(&mut self.phase, &mut self.seg_remaining) {
                self.state = DashState::Working;
            }
            if stashing && self.state == DashState::Working && self.phase.active {
                self.dense = Some(Solid::Stashed);
            } else {
                self.get_input();
            }
        } else if self.phase.remaining < self.seg_remaining {
            // next transition is a dash transition
            let seg = self.current_seg.subsegment(self.t..1.0);
            let t1 = seg.inv_arclen(self.phase.remaining, DASH_ACCURACY);
            if self.phase.active {
                let subseg = seg.subsegment(0.0..t1);
                result = Some(crate::cull::seg_to_el(&subseg));
                self.state = DashState::Working;
            } else {
                let p = seg.eval(t1);
                result = Some(PathEl::MoveTo(p));
            }
            self.t += t1 * (1.0 - self.t);
            self.seg_remaining -= self.phase.remaining;
            self.pattern.next(&mut self.phase);
        } else {
            if self.phase.active {
                let seg = self.current_seg.subsegment(self.t..1.0);
                result = Some(crate::cull::seg_to_el(&seg));
            }
            self.phase.remaining -= self.seg_remaining;
            self.get_input();
        }
        result
    }

    fn handle_closepath(&mut self) {
        if self.state == DashState::ToStash {
            // Have looped back without breaking a dash, just play it back
            self.stash.push(PathEl::ClosePath);
        } else if self.phase.active {
            // connect with path in stash, skip MoveTo.
            self.stash_ix = 1;
        }
        self.state = DashState::FromStash;
        self.reset_phase();
    }

    fn reset_phase(&mut self) {
        self.phase = self.pattern.start();
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

    /// A hidden line 2^54 - 200 long before one through the view.
    /// Its length is 4 modulo the period, so the visible line starts at phase 4;
    /// subtracting the dash before the modulo would round that to 3.
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

    /// A source of pseudo-random numbers.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn number(&mut self, low: f64, high: f64) -> f64 {
            low + (self.next() >> 11) as f64 / (1_u64 << 53) as f64 * (high - low)
        }
    }

    /// A path of subpaths of lines and, unless `lines` is set, curves, some closed,
    /// some drawn on from where a closed one starts, some closed again at once.
    fn random_path(random: &mut Random, lines: bool) -> BezPath {
        let point = |random: &mut Random| (random.number(0.0, 400.0), random.number(0.0, 400.0));
        let mut path = BezPath::new();
        path.move_to(point(random));
        for _ in 0..2 + random.next() % 10 {
            match random.next() % 7 {
                2 if !lines => {
                    let (control, end) = (point(random), point(random));
                    path.quad_to(control, end);
                }
                3 if !lines => {
                    let (first, second, end) = (point(random), point(random), point(random));
                    path.curve_to(first, second, end);
                }
                4 => path.move_to(point(random)),
                5 | 6 => path.close_path(),
                _ => path.line_to(point(random)),
            }
        }
        path
    }

    fn random_pattern(random: &mut Random) -> (Vec<f64>, f64) {
        let entries = (0..1 + random.next() % 5)
            .map(|_| match random.next() % 4 {
                0 => 0.0,
                _ => random.number(0.5, 60.0),
            })
            .collect();
        let offset = match random.next() % 3 {
            0 => 0.0,
            1 => random.number(-100.0, 100.0),
            _ => random.number(-1e6, 1e6),
        };
        (entries, offset)
    }

    /// The segments of `path` as dashing walks them.
    fn segments(path: &BezPath) -> Vec<PathSeg> {
        let mut subpaths = Subpaths::default();
        path.iter()
            .filter_map(|el| match subpaths.walk(el) {
                Walk::Segment(seg) | Walk::Close(Some(seg)) => Some(seg),
                Walk::Move | Walk::Close(None) => None,
            })
            .collect()
    }

    /// `phases` gives each segment of the path with the phase dashing walks it from,
    /// wherever subpaths start and close, and segments are drawn on from where a
    /// closed subpath starts. A closing that walks no line can make dashing walk a
    /// segment again or join a dash to the one drawn last, which the test below
    /// covers.
    #[test]
    fn phases_are_those_dashing_walks() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let mut checked = 0;
        for case in 0..1000 {
            let path = random_path(&mut random, false);
            let (entries, offset) = random_pattern(&mut random);
            if entries.iter().sum::<f64>() < 1.0 {
                continue;
            }
            let pattern = Pattern::new(offset, &entries);
            let walked: Vec<(PathSeg, Phase)> = phases(path.iter(), &pattern).collect();
            let mut subpaths = Subpaths::default();
            if path
                .iter()
                .any(|el| matches!(subpaths.walk(el), Walk::Close(None)))
            {
                continue;
            }
            let segments = segments(&path);
            assert!(
                walked
                    .iter()
                    .map(|(seg, _)| *seg)
                    .eq(segments.iter().copied())
            );
            let out = dash(
                path.iter().map(|el| (el, false)),
                offset,
                &entries,
                Affine::IDENTITY,
            );
            let mut flat = Vec::new();
            crate::kurbo::flatten(out, 1e-5, |el| flat.push(el));
            let (lines, ends) = dash_lines(&flat);
            for (index, (seg, phase)) in walked.iter().enumerate() {
                let length = seg.arclen(DASH_ACCURACY);
                // Where another segment passes, its dashes could be there.
                let elsewhere = |p: Point| {
                    segments.iter().enumerate().any(|(other, seg)| {
                        other != index && seg.nearest(p, 1e-9).distance_sq < 1e-4
                    })
                };
                for step in 1..40 {
                    let at = length * f64::from(step) / 40.0;
                    let p = seg.eval(seg.inv_arclen(at, 1e-9));
                    if elsewhere(p) || ends.iter().any(|end| end.distance(p) < 1e-2) {
                        continue;
                    }
                    checked += 1;
                    let (start, end, drawn) = pattern.entry_around(*phase, at);
                    assert!(start <= at && at <= end, "case {case}");
                    assert_eq!(
                        on_dash_within(&lines, p, 1e-3),
                        drawn,
                        "case {case}, segment {index} at {at}: {}",
                        path.to_svg()
                    );
                    // The dashes around a gap end and start where it says.
                    for stop in [start, end] {
                        let q = seg.eval(seg.inv_arclen(stop.clamp(0.0, length), 1e-9));
                        if !drawn && (0.0..=length).contains(&stop) && !elsewhere(q) {
                            assert!(on_dash_within(&lines, q, 1e-3), "case {case}");
                        }
                    }
                }
            }
        }
        assert!(checked > 50_000, "{checked} points checked");
    }

    /// Every dash, also where a closing walks a segment again, starts and ends at an
    /// end of a segment or of an entry of a segment `phases` gives.
    #[test]
    fn phases_give_every_end_of_a_dash() {
        let mut random = Random(0x2545_f491_4f6c_dd1d);
        let (mut checked, mut again) = (0, 0);
        for case in 0..2000 {
            let path = random_path(&mut random, true);
            let (entries, offset) = random_pattern(&mut random);
            if entries.iter().sum::<f64>() < 1.0 {
                continue;
            }
            let pattern = Pattern::new(offset, &entries);
            let walked: Vec<(PathSeg, Phase)> = phases(path.iter(), &pattern).collect();
            again += usize::from(walked.len() > segments(&path).len());
            let out = dash(
                path.iter().map(|el| (el, false)),
                offset,
                &entries,
                Affine::IDENTITY,
            );
            for el in out {
                let (PathEl::MoveTo(q) | PathEl::LineTo(q)) = el else {
                    continue;
                };
                checked += 1;
                let found = walked.iter().any(|(seg, phase)| {
                    let PathSeg::Line(line) = *seg else {
                        unreachable!("lines only")
                    };
                    let length = line.length();
                    if line.p0.distance(q) < 1e-6 || line.p1.distance(q) < 1e-6 {
                        return true;
                    }
                    let at = (q - line.p0).dot(line.p1 - line.p0) / length;
                    if !(0.0..=length).contains(&at) || line.eval(at / length).distance(q) > 1e-6 {
                        return false;
                    }
                    let (start, end, _) = pattern.entry_around(*phase, at);
                    (at - start).abs() < 1e-5 || (end - at).abs() < 1e-5
                });
                assert!(found, "case {case}: {q:?} of {}", path.to_svg());
            }
        }
        assert!(
            checked > 20_000 && again > 100,
            "{checked} ends, {again} paths walked again"
        );
    }

    /// The lines of `out`, dashes flattened to lines, and where each dash starts and
    /// ends.
    fn dash_lines(out: &[PathEl]) -> (Vec<Line>, Vec<Point>) {
        let (mut lines, mut ends) = (Vec::new(), Vec::new());
        let (mut start, mut last) = (Point::ORIGIN, Point::ORIGIN);
        for el in out {
            match *el {
                PathEl::MoveTo(q) => {
                    ends.extend([last, q]);
                    (start, last) = (q, q);
                }
                PathEl::LineTo(q) => {
                    lines.push(Line::new(last, q));
                    last = q;
                }
                PathEl::ClosePath => {
                    lines.push(Line::new(last, start));
                    last = start;
                }
                _ => panic!("lines only: {el:?}"),
            }
        }
        ends.push(last);
        (lines, ends)
    }

    fn on_dash_within(lines: &[Line], p: Point, distance: f64) -> bool {
        lines
            .iter()
            .any(|l| l.nearest(p, 1e-9).distance_sq < distance * distance)
    }
}
