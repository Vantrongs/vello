// Copyright 2023 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dashing as in `kurbo::dash` (kurbo 0.13.1 `stroke.rs`, `DashIterator`), copied so
//! its work stays bounded by the view:
//!
//! - A piece of a segment outside the view (flagged by `cull::SplitHuge`) is replaced
//!   by its chord, drawn solid: its whole dash periods are skipped in one step, and
//!   it paints nothing in view whatever its dashes. Kurbo walks every dash, so a
//!   dashed segment with huge coordinates took time in proportion to its length.
//! - A segment along which walking the pattern would take more than
//!   `DENSE_ENTRIES_PER_PX` entries per device pixel, or more than the `PATH_ENTRIES`
//!   the path may still walk, is cut into windows of `WINDOW_PX` device pixels, and
//!   each window draws one dash holding the dashed length the pattern has in it
//!   (`Filter`): the dashes keep their ink and gaps wider than a window stay
//!   unpainted, at a cost set by the segment's device length. Kurbo emits every
//!   dash, so a pattern far finer than a pixel along a segment (a transform that
//!   compresses its direction, or many tiny entries) emitted dashes in proportion to
//!   the inverse of their device size.
//!
//! A pattern whose whole period is under half a device pixel at the transform's
//! largest stretch has no gap a pixel resolves; `flatten::stroke` strokes it solid
//! ([`unresolved`]), as `MuPDF` does (1.28 `draw-path.c`, `do_flatten_stroke`).
//!
//! Every other segment is dashed exactly as kurbo does.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{
    Affine, CubicBez, Line, ParamCurve, ParamCurveArclen, PathEl, PathSeg, Point, QuadBez,
};
use alloc::vec::Vec;

const DASH_ACCURACY: f64 = 1e-6;

/// Pattern entries (dashes and gaps) per device pixel along a segment beyond which
/// the segment is drawn in windows (`Filter`) instead of dash by dash. Entries, not
/// periods, are counted: each is an element out, and a period can hold any number.
const DENSE_ENTRIES_PER_PX: f64 = 32.0;

/// Pattern entries, or windows, one path may walk; segments beyond are drawn in
/// fewer windows, down to one per segment. Segments larger than the view are split
/// and their hidden pieces skipped (`cull::SplitHuge`), so drawings stay far below
/// this; it bounds the work for a path whose geometry is numerically noisy
/// (coordinates near the limits of `f64` under a nearly singular transform), whose
/// split pieces can each look long on the device. A dash and its gap take about 64
/// bytes as a thin outline and 300 as an expanded one.
const PATH_ENTRIES: f64 = (1 << 22) as f64;

/// The device length of a window of a segment drawn in windows (`Filter`).
const WINDOW_PX: f64 = 0.25;

/// Whether `dashes` has no gap a device pixel resolves under a transform whose
/// largest stretch is `scale`: its whole period is under half a pixel there (or it is
/// empty of length). Such a stroke is drawn solid, as `MuPDF` does.
pub(crate) fn unresolved(dashes: &[f64], scale: f64) -> bool {
    dashes.iter().sum::<f64>() * scale < 0.5
}

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
        solid: None,
        filter: None,
        pattern: None,
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
    /// Whether `current_seg` is a hidden piece drawn solid (`set_segment`), and if so
    /// how far it is out.
    solid: Option<Solid>,
    /// Whether `current_seg` is drawn in windows (`set_segment`): then `is_active` and
    /// `dash_remaining` describe its windows' dashes, not the pattern.
    filter: Option<Filter>,
    /// The pattern's prefix sums, made when a segment is first drawn in windows.
    pattern: Option<Pattern>,
    affine: Affine,
    /// Pattern entries (or windows) the path may still walk (`PATH_ENTRIES`).
    entries_left: f64,
}

/// How far a hidden piece drawn solid is out.
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
    /// dense to walk one by one (see the module) is drawn in windows.
    fn set_segment(&mut self, seg: PathSeg, skip: bool) {
        self.filter = None;
        if skip {
            // Relative accuracy: kurbo's absolute 1e-6 on a huge segment recurses to
            // its depth limit for nothing.
            let len = seg.arclen(DASH_ACCURACY.max(1e-9 * control_length(&seg)));
            self.solid = Some(Solid::Start);
            self.seg_remaining = self.less_whole_periods(len);
            self.current_seg = PathSeg::Line(Line::new(seg.start(), seg.end()));
            return;
        }
        self.solid = None;
        let len = seg.arclen(DASH_ACCURACY);
        self.current_seg = seg;
        self.seg_remaining = len;
        let entries = len / self.period * self.entries_per_period;
        if self.period.is_nan() || self.period <= 0.0 {
            return;
        }
        let device_length = self.device_length(&seg);
        if entries > self.entries_left || entries > DENSE_ENTRIES_PER_PX * (device_length + 1.0) {
            let windows = (device_length / WINDOW_PX)
                .ceil()
                .min(self.entries_left)
                .max(1.0);
            self.entries_left = (self.entries_left - windows).max(0.0);
            self.start_filter(len, windows);
        } else {
            self.entries_left -= entries;
        }
    }

    /// Draws the current segment, `len` long, in `windows` windows.
    #[cold]
    fn start_filter(&mut self, len: f64, windows: f64) {
        let pattern = self
            .pattern
            .get_or_insert_with(|| Pattern::new(self.dashes));
        let entry = pattern.entry_of(self.dash_ix, self.is_active, self.dashes.len());
        let start = (pattern.starts[entry + 1] - self.dash_remaining).clamp(0.0, pattern.period());
        let mut filter = Filter {
            window: len / windows,
            windows,
            next: 0.0,
            second: false,
            start,
            dashing_at_start: self.is_active,
            len,
            laid_out: (-1.0, false, 0.0, 0.0),
        };
        self.dash_remaining = filter.run(pattern, self.is_active);
        self.filter = Some(filter);
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
        } else if let Some(solid) = self.solid {
            if solid == Solid::Stashed {
                let end = self.current_seg.end();
                self.get_input();
                return Some(PathEl::MoveTo(end));
            }
            if solid == Solid::Start && !self.is_active {
                // A dash starts with the segment.
                self.solid = Some(Solid::Moved);
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
                self.solid = Some(Solid::Stashed);
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
            if let (Some(filter), Some(pattern)) = (&mut self.filter, &self.pattern) {
                self.dash_remaining = filter.run(pattern, self.is_active);
            } else {
                self.dash_ix += 1;
                if self.dash_ix == self.dashes.len() {
                    self.dash_ix = 0;
                }
                self.dash_remaining = self.dashes[self.dash_ix];
            }
        } else {
            if self.is_active {
                let seg = self.current_seg.subsegment(self.t..1.0);
                result = Some(crate::cull::seg_to_el(&seg));
            }
            if self.filter.is_some() {
                if let Some(el) = self.end_filter() {
                    result = Some(el);
                }
            } else {
                self.dash_remaining -= self.seg_remaining;
            }
            self.get_input();
        }
        result
    }

    /// Ends a segment drawn in windows: the dash state becomes the pattern's at the
    /// segment's end. Where the windows' dash and the pattern's state differ there, the
    /// dash ends (as a dash ending in a gap does), or a move starts the pattern's dash
    /// at the end.
    #[cold]
    fn end_filter(&mut self) -> Option<PathEl> {
        let (Some(filter), Some(pattern)) = (self.filter.take(), &self.pattern) else {
            return None;
        };
        let drawing = self.is_active;
        let at = (filter.start + filter.len).rem_euclid(pattern.period());
        let entry = pattern.entry_at(at);
        self.dash_ix = entry % self.dashes.len();
        self.is_active = entry % 2 == 0;
        self.dash_remaining = pattern.starts[entry + 1] - at;
        if drawing && !self.is_active {
            self.state = DashState::Working;
        }
        (!drawing && self.is_active).then(|| PathEl::MoveTo(self.current_seg.end()))
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

/// The pattern as the walk repeats it (an odd-length array twice), with prefix sums,
/// for the dash state and the dashed length at any distance into a period.
struct Pattern {
    /// Where each entry starts, then the period.
    starts: Vec<f64>,
    /// The dashed length before each entry, then the period's.
    dashed: Vec<f64>,
}

impl Pattern {
    fn new(dashes: &[f64]) -> Self {
        let entries = if dashes.len() % 2 == 1 {
            2 * dashes.len()
        } else {
            dashes.len()
        };
        let mut starts = Vec::with_capacity(entries + 1);
        let mut dashed = Vec::with_capacity(entries + 1);
        let (mut at, mut on) = (0.0, 0.0);
        for i in 0..entries {
            starts.push(at);
            dashed.push(on);
            let d = dashes[i % dashes.len()];
            at += d;
            if i % 2 == 0 {
                on += d;
            }
        }
        starts.push(at);
        dashed.push(on);
        Self { starts, dashed }
    }

    fn period(&self) -> f64 {
        self.starts[self.starts.len() - 1]
    }

    /// The entry the walk is in at `dash_ix` of `len` entries, dashing or not: an
    /// odd-length array's second pass has the other parity.
    fn entry_of(&self, dash_ix: usize, is_active: bool, len: usize) -> usize {
        if dash_ix.is_multiple_of(2) == is_active {
            dash_ix
        } else {
            dash_ix + len
        }
    }

    /// The entry at `at` in `[0, period)`: the last that starts at or before it.
    fn entry_at(&self, at: f64) -> usize {
        let entries = self.starts.len() - 1;
        self.starts[..entries]
            .partition_point(|&s| s <= at)
            .saturating_sub(1)
    }

    /// The dashed length in `[0, at)`, `at` in `[0, period]`.
    fn dashed_before(&self, at: f64) -> f64 {
        let i = self.entry_at(at);
        let mut d = self.dashed[i];
        if i.is_multiple_of(2) {
            d += (at - self.starts[i]).clamp(0.0, self.starts[i + 1] - self.starts[i]);
        }
        d
    }

    /// The dashed length in `width` from `at` in `[0, period)`.
    fn dashed(&self, at: f64, width: f64) -> f64 {
        let period = self.period();
        let whole = (width / period).floor();
        let end = at + (width - whole * period);
        let total = self.dashed[self.dashed.len() - 1];
        let part = if end <= period {
            self.dashed_before(end) - self.dashed_before(at)
        } else {
            total - self.dashed_before(at) + self.dashed_before(end - period)
        };
        whole * total + part
    }
}

/// A segment drawn in windows. Each window draws one dash holding the dashed length
/// the pattern has in it: first if the pattern dashes at the window's start, else
/// last, so the dashes of consecutive windows join where the pattern runs on, and a
/// gap wider than a window stays a gap. Consecutive parts of the same kind form one
/// run, a dash or a gap, which the walk steps over like a pattern entry.
struct Filter {
    /// The arc length of a window (the last may be shorter).
    window: f64,
    /// The number of windows.
    windows: f64,
    /// The window the walk is in, and whether in its second part.
    next: f64,
    second: bool,
    /// Where in the period the segment starts, whether the walk dashes there, and the
    /// segment's arc length.
    start: f64,
    dashing_at_start: bool,
    len: f64,
    /// The last window laid out: its index and `layout`.
    laid_out: (f64, bool, f64, f64),
}

impl Filter {
    /// Whether window `k` starts with its dash, and the lengths of its two parts.
    fn layout(&mut self, pattern: &Pattern, k: f64) -> (bool, f64, f64) {
        if self.laid_out.0 == k {
            return (self.laid_out.1, self.laid_out.2, self.laid_out.3);
        }
        let from = k * self.window;
        let width = (self.len - from).clamp(0.0, self.window);
        let at = (self.start + from).rem_euclid(pattern.period());
        let dash_first = if k == 0.0 {
            self.dashing_at_start
        } else {
            pattern.entry_at(at).is_multiple_of(2)
        };
        let ink = pattern.dashed(at, width).clamp(0.0, width);
        let first = if dash_first { ink } else { width - ink };
        self.laid_out = (k, dash_first, first, width - first);
        (dash_first, first, width - first)
    }

    /// The length of the run of dashes (`dash`) or gaps from where the walk is, which
    /// it then steps past; infinite if the run reaches the segment's end.
    fn run(&mut self, pattern: &Pattern, dash: bool) -> f64 {
        let mut len = 0.0;
        while self.next < self.windows {
            let (dash_first, a, b) = self.layout(pattern, self.next);
            let (kind, part) = if self.second {
                (!dash_first, b)
            } else {
                (dash_first, a)
            };
            if part > 0.0 && kind != dash {
                return len;
            }
            len += part;
            if self.second {
                self.next += 1.0;
            }
            self.second = !self.second;
        }
        f64::INFINITY
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
    use alloc::vec;

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

    /// The dashed length of horizontal `out` (lines only) in each unit of x from
    /// `x0`, `n` units, x scaled by `scale` to the device.
    fn ink_per_pixel(out: &[PathEl], x0: f64, scale: f64, n: usize) -> Vec<f64> {
        let mut ink = vec![0.0; n];
        let mut last = Point::ORIGIN;
        for el in out {
            match *el {
                PathEl::MoveTo(p) => last = p,
                PathEl::LineTo(p) => {
                    let (a, b) = ((last.x - x0) * scale, (p.x - x0) * scale);
                    let (a, b) = (a.min(b), a.max(b));
                    for (i, v) in ink.iter_mut().enumerate() {
                        let (l, r) = (i as f64, i as f64 + 1.0);
                        *v += (b.min(r) - a.max(l)).max(0.0);
                    }
                    last = p;
                }
                PathEl::ClosePath => {}
                _ => panic!("lines only: {el:?}"),
            }
        }
        ink
    }

    /// Dashing `line` with `dashes` under `affine` (device x = `scale` path x).
    fn dash_line(line: Line, dashes: &[f64], affine: Affine) -> Vec<PathEl> {
        let input = [
            (PathEl::MoveTo(line.p0), false),
            (PathEl::LineTo(line.p1), false),
        ];
        starts_with_a_move(dash(input.into_iter(), 0.0, dashes, affine).collect())
    }

    /// The `dash-mixed` probe: a wide dash and gap, then 10 000 entries of 5e-6, at
    /// 2 px per unit. Walking it takes 250 entries per pixel, so it is drawn in
    /// windows; the 20 px gap stays unpainted and each pixel keeps the ink the exact
    /// dashes give it, up to a window's worth (a quarter pixel).
    #[test]
    fn a_dense_pattern_keeps_its_wide_gaps_and_ink() {
        let mut dashes = vec![10.0, 10.0];
        dashes.extend(core::iter::repeat_n(5e-6, 10_000));
        let line = Line::new((0.0, 25.0), (50.0, 25.0));
        let affine = Affine::scale(2.0);
        let a = dash_line(line, &dashes, affine);
        assert!(a.len() < 2 * 400 + 4, "{} elements", a.len());
        let exact = kurbo(
            &BezPath::from_vec(vec![PathEl::MoveTo(line.p0), PathEl::LineTo(line.p1)]),
            0.0,
            &dashes,
        );
        let (ia, ie) = (
            ink_per_pixel(&a, 0.0, 2.0, 100),
            ink_per_pixel(&exact, 0.0, 2.0, 100),
        );
        for x in 0..100 {
            assert!(
                (ia[x] - ie[x]).abs() <= 0.25 + 1e-9,
                "pixel {x}: {} vs {}",
                ia[x],
                ie[x]
            );
        }
        assert!(ia[20..40].iter().all(|&v| v == 0.0), "{:?}", &ia[20..40]);
        assert!(
            ia[..20].iter().all(|&v| (v - 1.0).abs() < 1e-9),
            "{:?}",
            &ia[..20]
        );
        let total = |v: &[f64]| v.iter().sum::<f64>();
        assert!((total(&ia) - total(&ie)).abs() < 1e-6);
    }

    /// Windows that do not line up with pixels move ink by at most a window; a closed
    /// subpath drawn in windows keeps its ink (its first dash held back for the
    /// closing one, as kurbo does).
    #[test]
    fn windows_keep_the_ink_of_each_pixel() {
        let mut dashes = vec![5.0, 3.0];
        dashes.extend(core::iter::repeat_n(0.001, 1000));
        let line = Line::new((0.37, 10.0), (97.0, 10.0));
        let a = dash_line(line, &dashes, Affine::IDENTITY);
        let path = BezPath::from_vec(vec![PathEl::MoveTo(line.p0), PathEl::LineTo(line.p1)]);
        let exact = kurbo(&path, 0.0, &dashes);
        let (ia, ie) = (
            ink_per_pixel(&a, 0.0, 1.0, 100),
            ink_per_pixel(&exact, 0.0, 1.0, 100),
        );
        for x in 0..100 {
            assert!(
                (ia[x] - ie[x]).abs() <= 0.25 + 1e-9,
                "pixel {x}: {} vs {}",
                ia[x],
                ie[x]
            );
        }
        let mut square = BezPath::new();
        square.move_to((10.0, 10.0));
        square.line_to((90.0, 10.0));
        square.line_to((90.0, 90.0));
        square.line_to((10.0, 90.0));
        square.close_path();
        for offset in [0.0, 6.0] {
            let a = mine(&square, offset, &dashes, VIEW);
            let b = kurbo(&square, offset, &dashes);
            assert!(a.len() < 4 * 2 * 330, "{} elements", a.len());
            let length = |out: &[PathEl]| {
                let (mut last, mut sum) = (Point::ORIGIN, 0.0);
                for el in out {
                    match *el {
                        PathEl::MoveTo(p) => last = p,
                        PathEl::LineTo(p) => {
                            sum += last.distance(p);
                            last = p;
                        }
                        _ => {}
                    }
                }
                sum
            };
            assert!((length(&a) - length(&b)).abs() < 1e-6, "{offset}");
        }
    }

    /// A segment whose pattern is far below a pixel on the device, though not at the
    /// transform's largest stretch, is drawn in windows that keep its ink, and the
    /// next one starts at the phase walking it would leave.
    #[test]
    fn dense_dashes_keep_their_ink_and_phase() {
        let dashes = [3.0, 7.0];
        // x is compressed 1000-fold: the first line, 1000.3 long in path space (100
        // periods), is 1 px on the device; the second, vertical, is not compressed.
        let affine = Affine::scale_non_uniform(1e-3, 1.0);
        let mut p = BezPath::new();
        p.move_to((-1000.3, 50.0));
        p.line_to((0.0, 50.0));
        p.line_to((0.0, 100.0));
        let wide = Cull::new(affine, [-1e6, -1e6, 1e6, 1e6]);
        for offset in [1.5, 4.0] {
            let split = SplitHuge::new(p.iter(), wide, true);
            let a = starts_with_a_move(dash(split, offset, &dashes, affine).collect());
            let mut first = BezPath::new();
            first.move_to((-1000.3, 50.0));
            first.line_to((0.0, 50.0));
            let exact = kurbo(&first, offset, &dashes);
            // Its ink, in the five quarter-pixel windows the first line is cut into.
            let horizontal: Vec<PathEl> = {
                let (mut out, mut last) = (Vec::new(), Point::ORIGIN);
                for el in &a {
                    match *el {
                        PathEl::MoveTo(q) => last = q,
                        PathEl::LineTo(q) => {
                            if last.y == 50.0 && q.y == 50.0 {
                                out.extend([PathEl::MoveTo(last), PathEl::LineTo(q)]);
                            }
                            last = q;
                        }
                        _ => {}
                    }
                }
                out
            };
            assert!(horizontal.len() <= 2 * 6, "{a:?}");
            let window = 1000.3 / 5.0;
            let (ia, ie) = (
                ink_per_pixel(&horizontal, -1000.3, 1.0 / window, 5),
                ink_per_pixel(&exact, -1000.3, 1.0 / window, 5),
            );
            for k in 0..5 {
                assert!(
                    (ia[k] - ie[k]).abs() < 1e-9,
                    "{offset} window {k}: {ia:?} {ie:?}"
                );
            }
            // The vertical line is dashed from phase (offset + 1000.3) mod 10, as kurbo
            // dashes the whole path.
            let b = kurbo(&p, offset, &dashes);
            same_dashes_in_view(&a, &b, Line::new((0.0, 50.0), (0.0, 100.0)), 1e-6);
        }
    }

    /// The whole period under half a device pixel: no gap is resolved.
    #[test]
    fn unresolved_patterns() {
        assert!(unresolved(&[0.1, 0.1], 2.0));
        assert!(!unresolved(&[0.125, 0.125], 2.0));
        assert!(unresolved(&[0.0, 0.0], 1.0));
        let mut mixed = vec![10.0, 10.0];
        mixed.extend(core::iter::repeat_n(5e-6, 10_000));
        assert!(!unresolved(&mixed, 2.0));
    }
}
