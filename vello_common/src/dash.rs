// Copyright 2023 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Dashing as in `kurbo::dash` (kurbo 0.13.1 `stroke.rs`, `DashIterator`), copied so
//! its work stays bounded by the view:
//!
//! - Outside the cap margin, only the joins at a piece's endpoints can
//!   reach the view. Skip its interior and preserve the phase and tangents.
//! - Runs of subpixel entries are coalesced. A whole pattern can also become
//!   subpixel along a compressed direction. Skipped lengths retain their phase.
//!
//! Every other segment is dashed exactly as kurbo does.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{
    Affine, CubicBez, Line, ParamCurve, ParamCurveArclen, PathEl, PathSeg, Point, QuadBez,
};
use crate::path_budget::Meter;
use alloc::vec::Vec;

const DASH_ACCURACY: f64 = 1e-6;

const DENSE_ENTRIES_PER_PX: f64 = 32.0;

#[derive(Clone, Copy)]
struct Entry {
    length: f64,
    end: f64,
    active: bool,
}

/// Dashes `inner` like `kurbo::dash`, skipping the dash periods of the segments
/// flagged `true`; `affine` maps the path to device space.
pub(crate) fn dash<'a, T: Iterator<Item = (PathEl, bool)>>(
    inner: T,
    dash_offset: f64,
    dashes: &[f64],
    affine: Affine,
    paints_dots: bool,
    meter: Meter<'a>,
) -> DashIterator<'a, T> {
    // Ensure that offset is positive and minimal by normalization using period
    let period: f64 = dashes.iter().sum();
    // The SVG spec requires odd-length dash arrays to be doubled to become even-length:
    // <https://www.w3.org/TR/SVG2/painting.html#StrokeDasharrayProperty>
    // This prevents gaps and dashes from swapping with one another as the offset increases.
    let period = if dashes.len() % 2 == 1 {
        2.0 * period
    } else {
        period
    };
    let valid = period.is_finite()
        && period > 0.0
        && dash_offset.is_finite()
        && dashes.iter().all(|&d| d.is_finite() && d >= 0.0);
    let dashes = if valid { dashes } else { &[] };
    let entries = if dashes.len() % 2 == 1 {
        2 * dashes.len()
    } else {
        dashes.len()
    };

    // Start at a painted dash so gaps meeting across the period boundary are
    // adjacent in the iterator. Restore the original phase below.
    let first = if paints_dots {
        Some(0)
    } else {
        (0..entries)
            .step_by(2)
            .find(|i| dashes[i % dashes.len()] > 0.)
    };
    let first_ix = first.unwrap_or(0);
    let phase_shift: f64 = dashes.iter().cycle().take(first_ix).sum();
    let mut raw = (first_ix..entries)
        .chain(0..first_ix)
        .filter_map(|ix| {
            let length = dashes[ix % dashes.len()];
            let active = ix.is_multiple_of(2);
            (!active || length > 0. || paints_dots).then_some((length, active))
        })
        .peekable();
    let mut max_entry = 0.0_f64;
    let mut runs = core::iter::from_fn(|| {
        let (mut length, active) = raw.next()?;
        if !active {
            while let Some((gap, _)) = raw.next_if(|(_, active)| !active) {
                length += gap;
            }
        }
        max_entry = max_entry.max(length);
        Some((length, active))
    })
    .peekable();

    let scale = crate::flatten::max_scale(affine);
    let tiny = |length: f64| length * scale < 1. / DENSE_ENTRIES_PER_PX;
    let mut pattern: Vec<Entry> = Vec::new();
    let mut end = 0.;
    while let Some((mut length, mut active)) = runs.next() {
        if tiny(length) {
            while let Some((next, paints)) = runs.next_if(|(next, _)| tiny(*next)) {
                length += next;
                active |= paints;
            }
        }
        if !meter.room(&mut pattern, 1) {
            break;
        }
        end += length;
        pattern.push(Entry {
            length,
            end,
            active,
        });
    }
    let period = end;
    let (dash_ix, dash_remaining) = if meter.failed() || pattern.is_empty() {
        (0, 0.)
    } else {
        let offset = dash_offset.rem_euclid(period) - phase_shift;
        if offset < 0. {
            // The phase lies in the gap wrapped to the end. Adding a long
            // period here could round the remaining gap away.
            (pattern.len() - 1, -offset)
        } else {
            let ix = pattern.partition_point(|e| e.end < offset);
            (ix, pattern[ix].end - offset)
        }
    };
    let is_active = pattern.get(dash_ix).is_some_and(|e| e.active);
    DashIterator {
        inner,
        meter,
        input_done: valid && first.is_none(),
        closepath_pending: false,
        dashes: pattern,
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
        max_entry,
        dense: None,
        hidden: false,
        hidden_crossed: false,
        hidden_tail: None,
        hidden_move: None,
        affine,
    }
}

/// An implementation of dashing as an iterator-to-iterator transformation.
pub(crate) struct DashIterator<'a, T> {
    meter: Meter<'a>,
    inner: T,
    input_done: bool,
    closepath_pending: bool,
    dashes: Vec<Entry>,
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
    max_entry: f64,
    /// Whether `current_seg` is drawn solid (`set_segment`), and if so how far it is
    /// out.
    dense: Option<Solid>,
    hidden: bool,
    hidden_crossed: bool,
    hidden_tail: Option<Point>,
    hidden_move: Option<Point>,
    affine: Affine,
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
        if self.meter.failed() {
            return None;
        }
        if self.dashes.is_empty() {
            return self.inner.next().map(|(el, _)| el);
        }
        loop {
            if self.meter.failed() {
                return None;
            }
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
                        if !self.meter.room(&mut self.stash, 1) {
                            return None;
                        }
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

    fn set_segment(&mut self, seg: PathSeg, skip: bool) {
        self.hidden = skip;
        if skip {
            // Relative accuracy: kurbo's absolute 1e-6 on a huge segment recurses to
            // its depth limit for nothing.
            let extent = control_length(&seg);
            // Kurbo's curve arclength uses products of squared coordinates.
            let len = if extent > 1e70 && extent.is_finite() && !matches!(seg, PathSeg::Line(_)) {
                let normalized = Affine::scale(extent.recip()) * seg;
                normalized.arclen(1e-9) * extent
            } else {
                arc_length(&seg, DASH_ACCURACY.max(1e-9 * extent))
            };
            self.hidden_crossed = len > self.dash_remaining;
            self.seg_remaining = self.less_whole_periods(len);
            if let PathSeg::Line(line) = seg {
                let axis = if line.p0.x == line.p1.x {
                    Some((line.p0.y, line.p1.y))
                } else if line.p0.y == line.p1.y {
                    Some((line.p0.x, line.p1.x))
                } else {
                    None
                };
                if let Some((a, b)) = axis {
                    let (main, tail) = crate::expansion::two_sum(b, -a);
                    if tail != 0.0 && len > self.period + self.dash_remaining {
                        let phase = ((main.abs() % self.period)
                            + (main.signum() * tail % self.period)
                            - self.dash_remaining)
                            .rem_euclid(self.period);
                        self.seg_remaining = self.dash_remaining + phase;
                    }
                }
            }
            self.current_seg = seg;
            return;
        }
        let len = arc_length(&seg, DASH_ACCURACY);
        self.current_seg = seg;
        let all_tiny = match seg {
            PathSeg::Line(_) => {
                self.max_entry < len / self.device_length(&seg) / DENSE_ENTRIES_PER_PX
            }
            _ => {
                self.max_entry * crate::flatten::max_scale(self.affine) < 1. / DENSE_ENTRIES_PER_PX
            }
        };
        // If even the largest transition rounds away, a full period cannot
        // advance the remaining length. Consume its phase in one step.
        if all_tiny || len - self.max_entry == len {
            self.dense = Some(Solid::Start);
            self.seg_remaining = self.less_whole_periods(len);
        } else {
            self.dense = None;
            self.seg_remaining = len;
        }
    }

    /// `len` less as many whole periods as leave it longer than the current dash, so
    /// walking it gives the dash state walking `len` would.
    fn less_whole_periods(&self, len: f64) -> f64 {
        let (dash, period) = (self.dash_remaining, self.period);
        if len - dash <= period {
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

    fn advance_phase(&mut self) {
        if self.seg_remaining > self.dash_remaining {
            let end = self.dashes[self.dash_ix].end;
            let advance = (self.seg_remaining - self.dash_remaining) % self.period;
            let phase = if advance > self.period - end {
                advance - (self.period - end)
            } else {
                end + advance
            };
            self.dash_ix = self.dashes.partition_point(|e| e.end < phase);
            let entry = self.dashes[self.dash_ix];
            self.dash_remaining = entry.end - phase;
            self.is_active = entry.active;
        } else {
            self.dash_remaining -= self.seg_remaining;
        }
    }

    // Only the joins at the ends of a hidden segment can reach past the cap
    // margin. Keep their tangents and whether an on interval meets each end.
    // Any caps on the disconnected tangent segments stay outside the view.
    fn hidden_step(&mut self) -> Option<PathEl> {
        if let Some(p) = self.hidden_move.take() {
            if self.hidden_tail.is_none() {
                self.get_input();
            }
            return Some(PathEl::MoveTo(p));
        }
        if let Some(p) = self.hidden_tail.take() {
            self.get_input();
            return Some(PathEl::LineTo(p));
        }
        let seg = self.current_seg;
        let active = self.is_active;
        let crossed = self.hidden_crossed;
        let first_length = self.dash_remaining;
        self.advance_phase();
        if !crossed {
            let result = active.then(|| crate::cull::seg_to_el(&seg));
            self.get_input();
            return result;
        }
        let points: &[Point] = match seg {
            PathSeg::Line(l) => &[l.p0, l.p1],
            PathSeg::Quad(q) => &[q.p0, q.p1, q.p2],
            PathSeg::Cubic(c) => &[c.p0, c.p1, c.p2, c.p3],
        };
        let start = seg.start();
        let end = seg.end();
        // Long artificial tangents would also give the downstream stroke
        // expander the length of the hidden segment rather than of the dash.
        let tangent = |origin: Point, control: Point, length: f64| {
            let v = control - origin;
            let norm = v.x.hypot(v.y);
            if norm <= length {
                control
            } else {
                origin + crate::kurbo::Vec2::new(v.x / norm, v.y / norm) * length
            }
        };
        let tail = if self.is_active {
            self.hidden_tail = Some(end);
            let control = points
                .iter()
                .rev()
                .copied()
                .find(|p| *p != end)
                .unwrap_or(end);
            tangent(
                end,
                control,
                self.dashes[self.dash_ix].length - self.dash_remaining,
            )
        } else {
            end
        };
        self.state = DashState::Working;
        if active {
            self.hidden_move = Some(tail);
            let control = points
                .iter()
                .copied()
                .find(|p| *p != start)
                .unwrap_or(start);
            Some(PathEl::LineTo(tangent(start, control, first_length)))
        } else {
            if self.hidden_tail.is_none() {
                self.get_input();
            }
            Some(PathEl::MoveTo(tail))
        }
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
        } else if self.hidden {
            return self.hidden_step();
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
            if self.seg_remaining > self.dash_remaining {
                self.state = DashState::Working;
            }
            self.advance_phase();
            if stashing && self.state == DashState::Working && self.is_active {
                self.dense = Some(Solid::Stashed);
            } else {
                self.get_input();
            }
        } else if self.dash_remaining < self.seg_remaining {
            // next transition is a dash transition
            let seg = self.current_seg.subsegment(self.t..1.0);
            let t1 = if matches!(seg, PathSeg::Line(_)) {
                self.dash_remaining / control_length(&seg)
            } else {
                seg.inv_arclen(self.dash_remaining, DASH_ACCURACY)
            };
            if self.is_active {
                let subseg = seg.subsegment(0.0..t1);
                result = Some(crate::cull::seg_to_el(&subseg));
                if !self.dashes[(self.dash_ix + 1) % self.dashes.len()].active {
                    self.state = DashState::Working;
                }
            } else {
                let p = seg.eval(t1);
                result = Some(PathEl::MoveTo(p));
            }
            self.t += t1 * (1.0 - self.t);
            self.seg_remaining -= self.dash_remaining;
            self.dash_ix += 1;
            if self.dash_ix == self.dashes.len() {
                self.dash_ix = 0;
            }
            self.dash_remaining = self.dashes[self.dash_ix].length;
            self.is_active = self.dashes[self.dash_ix].active;
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
            if !self.meter.room(&mut self.stash, 1) {
                return;
            }
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
    let dist = |a: Point, b: Point| {
        let v = a - b;
        // Preserve Kurbo's rounding for ordinary dash boundaries.
        let max = v.x.abs().max(v.y.abs());
        if (1e-150..1e150).contains(&max) {
            v.hypot()
        } else {
            v.x.hypot(v.y)
        }
    };
    match *seg {
        PathSeg::Line(l) => dist(l.p0, l.p1),
        PathSeg::Quad(q) => dist(q.p0, q.p1) + dist(q.p1, q.p2),
        PathSeg::Cubic(c) => dist(c.p0, c.p1) + dist(c.p1, c.p2) + dist(c.p2, c.p3),
    }
}

fn arc_length(seg: &PathSeg, accuracy: f64) -> f64 {
    if matches!(seg, PathSeg::Line(_)) {
        control_length(seg)
    } else {
        seg.arclen(accuracy)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn invalid_pattern_preserves_path() {
        let path = [
            PathEl::MoveTo(Point::ORIGIN),
            PathEl::LineTo(Point::new(10., 2.)),
        ];
        for pattern in [&[][..], &[0.], &[-1.], &[f64::NAN], &[f64::MAX, f64::MAX]] {
            assert!(
                dash(
                    path.map(|el| (el, false)).into_iter(),
                    0.,
                    pattern,
                    Affine::IDENTITY,
                    true,
                    Meter::default()
                )
                .eq(path)
            );
        }
    }
    use super::*;
    use crate::cull::{Cull, SplitHuge};
    use crate::kurbo::{Affine, BezPath, ParamCurveNearest};

    const VIEW: [f64; 4] = [0.0, 0.0, 100.0, 100.0];

    fn mine(path: &BezPath, offset: f64, dashes: &[f64], rect: [f64; 4]) -> Vec<PathEl> {
        let split = SplitHuge::new(path.iter(), Cull::new(Affine::IDENTITY, rect), true);
        starts_with_a_move(
            dash(
                split,
                offset,
                dashes,
                Affine::IDENTITY,
                true,
                Meter::default(),
            )
            .collect(),
        )
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
    fn visible_quadratic_stays_curved_and_bounded_before_dashing() {
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.quad_to((1e80, 0.), (0., 1e160));
        let pieces: Vec<_> =
            SplitHuge::new(path.iter(), Cull::new(Affine::IDENTITY, VIEW), true).collect();
        assert!(pieces.len() < 1024);
        let mut curved = false;
        for &(el, hidden) in &pieces {
            if !hidden {
                let end = el.end_point().unwrap();
                assert!(end.x.abs() <= 200. && end.y.abs() <= 200.);
                curved |= matches!(el, PathEl::QuadTo(..));
            }
        }
        assert!(curved);
        let count = dash(
            pieces.into_iter(),
            0.,
            &[8., 8.],
            Affine::IDENTITY,
            false,
            Meter::default(),
        )
        .take(4096)
        .count();
        assert!((2..4096).contains(&count));
    }

    #[test]
    fn tiny_zero_dashes_share_one_gap_per_period() {
        let mut pattern = [0., 2_f64.powi(-30)].repeat(65536);
        pattern.extend([1., 1.]);
        let gap = 2_f64.powi(-14);
        let path = [
            PathEl::MoveTo(Point::ZERO),
            PathEl::LineTo(Point::new(65535., 0.)),
        ];
        let dashed = dash(
            path.map(|el| (el, false)).into_iter(),
            0.,
            &pattern,
            Affine::IDENTITY,
            false,
            Meter::default(),
        );
        assert_eq!(dashed.dashes.len(), 2);
        let mut start = 0.;
        let mut lines = 0;
        for el in dashed {
            match el {
                PathEl::MoveTo(p) => start = p.x,
                PathEl::LineTo(p) => {
                    assert!((start - (gap + f64::from(lines) * (2. + gap))).abs() < 1e-6);
                    assert!((p.x - (start + 1.).min(65535.)).abs() < 1e-6);
                    lines += 1;
                }
                _ => unreachable!(),
            }
        }
        assert_eq!(lines, ((65535. - gap) / (2. + gap)).ceil() as i32);
    }

    #[test]
    fn many_subpaths_retain_resolvable_dashes_after_long_hidden_prefixes() {
        // This pattern repeats [1, 2]. Its uncompressed period is large, but
        // every visible suffix has the same small explicit reference.
        let pattern = [1., 2.].repeat(131072);
        let subpaths = 16384;
        let input = (0..subpaths).flat_map(|row| {
            let y = f64::from(row);
            [
                (PathEl::MoveTo(Point::new(-300001., y)), false),
                (PathEl::LineTo(Point::new(0., y)), true),
                (PathEl::LineTo(Point::new(7., y)), false),
            ]
        });
        let mut expected = BezPath::new();
        expected.move_to((0., 0.));
        expected.line_to((7., 0.));
        let reference = kurbo(&expected, 1.25, &[1., 2.]);
        let (lines, _) = dashes_near_view(&reference);
        let expected: [bool; 7] =
            core::array::from_fn(|i| on_dash(&lines, Point::new(i as f64 + 0.5, 0.)));
        let mut rows = alloc::vec![[false;7]; subpaths as usize];
        let mut at = Point::ORIGIN;
        for el in dash(
            input,
            0.25,
            &pattern,
            Affine::IDENTITY,
            true,
            Meter::default(),
        ) {
            match el {
                PathEl::LineTo(p) => {
                    for (i, pixel) in rows[p.y as usize].iter_mut().enumerate() {
                        let x = i as f64 + 0.5;
                        *pixel |= at.x <= x && x < p.x;
                    }
                    at = p;
                }
                PathEl::MoveTo(p) => at = p,
                _ => unreachable!(),
            }
        }
        assert!(rows.into_iter().all(|row| row == expected));
    }

    #[test]
    fn resolvable_gaps_survive_high_cumulative_pattern_work() {
        let mut pattern = [1. / 1024.; 1024].to_vec();
        pattern.extend([64., 64.]);
        // At 129 pixels per subpath, walking the raw pattern would exceed
        // four million transitions. The following visible gap stays empty.
        let input = (0..8192).flat_map(|y| {
            let y = f64::from(y);
            [
                (PathEl::MoveTo(Point::new(0., y)), false),
                (PathEl::LineTo(Point::new(129., y)), false),
            ]
        });
        let mut at = Point::ORIGIN;
        let mut seen = [false; 8192];
        for el in dash(
            input,
            0.,
            &pattern,
            Affine::IDENTITY,
            true,
            Meter::default(),
        ) {
            match el {
                PathEl::MoveTo(p) => at = p,
                PathEl::LineTo(p) => {
                    assert!(!(at.x <= 100. && p.x > 100.));
                    seen[p.y as usize] |= at.x <= 32. && p.x > 32.;
                    at = p;
                }
                _ => unreachable!(),
            }
        }
        assert!(seen.into_iter().all(|covered| covered));
    }

    #[test]
    fn mixed_zero_entries_preserve_visible_gaps() {
        let mut line = BezPath::new();
        line.move_to((0., 0.));
        line.line_to((30., 0.));
        for pattern in [&[3., 0., 2., 4.][..], &[0., 3.]] {
            for offset in [0., 1., 17.] {
                assert_eq!(
                    mine(&line, offset, pattern, VIEW),
                    kurbo(&line, offset, pattern)
                );
            }
        }
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

    #[test]
    fn a_single_crossing_line_needs_constant_dash_work() {
        for exp in [12, 41, 82, 203, 900] {
            let mut path = BezPath::new();
            path.move_to((-2.0_f64.powi(exp), 50.));
            path.line_to((100., 50.));
            let pieces = || SplitHuge::new(path.iter(), Cull::new(Affine::IDENTITY, VIEW), true);
            assert!(pieces().count() <= 4, "2^{exp}");
            assert!(
                dash(
                    pieces(),
                    0.,
                    &[7., 3.],
                    Affine::IDENTITY,
                    true,
                    Meter::default()
                )
                .count()
                    <= 26,
                "2^{exp}"
            );
        }
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
        let a = starts_with_a_move(
            dash(
                input.into_iter(),
                0.0,
                &dashes,
                Affine::IDENTITY,
                true,
                Meter::default(),
            )
            .collect(),
        );
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

    #[test]
    fn clipping_preserves_straight_dash_density() {
        let mut line = BezPath::new();
        line.move_to((-4096., 10.));
        line.line_to((4096., 10.));
        let pieces = mine(&line, 0., &[0.031249; 2], VIEW);
        assert!(pieces.len() < 100, "{} elements", pieces.len());
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
        let a =
            starts_with_a_move(dash(split, 1.5, &dashes, affine, true, Meter::default()).collect());
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
        let a =
            starts_with_a_move(dash(split, 4.0, &dashes, affine, true, Meter::default()).collect());
        assert!(a.windows(2).any(|w| w == whole), "{a:?}");
        let b = kurbo(&p, 4.0, &dashes);
        same_dashes_in_view(&a, &b, Line::new((0.0, 50.0), (0.0, 100.0)), 1e-6);
    }

    #[test]
    fn phase_beyond_an_admitted_prefix_stops_the_iterator() {
        let budget = crate::path_budget::PathBudget::new(4 * size_of::<Entry>());
        let mut iter = dash(
            core::iter::empty(),
            4.5,
            &[1.; 8],
            Affine::IDENTITY,
            false,
            Meter::Raster(&budget),
        );
        assert_eq!(iter.dashes.len(), 4);
        assert!(iter.meter.failed());
        assert_eq!(iter.next(), None);
    }
}
