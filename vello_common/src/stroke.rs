// Copyright 2023 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Kurbo 0.13.1 stroke expansion with capacity checks on every path buffer.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{
    Affine, Arc, Cap, CubicBez, Join, PathEl, PathSeg, Point, QuadBez, Stroke, Vec2,
    common::solve_quadratic,
};
use crate::path_budget::Meter;
use alloc::vec::Vec;
use core::f64::consts::PI;

mod offset;

#[derive(Default, Debug)]
pub(crate) struct StrokeBuffers {
    output: Vec<PathEl>,
    forward: Vec<PathEl>,
    backward: Vec<PathEl>,
    result: Vec<PathEl>,
}

impl StrokeBuffers {
    pub(crate) fn expand(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        style: &Stroke,
        tolerance: f64,
        meter: Meter<'_>,
    ) {
        let mut ctx = StrokeCtx {
            output: Path::new(&mut self.output, meter),
            forward_path: Path::new(&mut self.forward, meter),
            backward_path: Path::new(&mut self.backward, meter),
            result_path: Path::new(&mut self.result, meter),
            start_pt: Point::ORIGIN,
            last_pt: Point::ORIGIN,
            start_norm: Vec2::ZERO,
            start_tan: Vec2::ZERO,
            last_tan: Vec2::ZERO,
            join_thresh: 2.0 * tolerance / style.width,
        };
        stroke_undashed(path, style, tolerance, &mut ctx);
        if meter.failed() {
            self.output.clear();
        }
    }

    pub(crate) fn output(&self) -> &[PathEl] {
        &self.output
    }
}

struct Path<'a> {
    data: &'a mut Vec<PathEl>,
    meter: Meter<'a>,
}

impl<'a> Path<'a> {
    fn new(data: &'a mut Vec<PathEl>, meter: Meter<'a>) -> Self {
        data.clear();
        Self { data, meter }
    }
    fn push(&mut self, el: PathEl) {
        if self.meter.room(self.data, 1) {
            self.data.push(el);
        }
    }
    fn move_to(&mut self, p: Point) {
        self.push(PathEl::MoveTo(p));
    }
    fn line_to(&mut self, p: Point) {
        self.push(PathEl::LineTo(p));
    }
    fn quad_to(&mut self, a: Point, b: Point) {
        self.push(PathEl::QuadTo(a, b));
    }
    fn curve_to(&mut self, a: Point, b: Point, c: Point) {
        self.push(PathEl::CurveTo(a, b, c));
    }
    fn close_path(&mut self) {
        self.push(PathEl::ClosePath);
    }
    fn extend(&mut self, els: &[PathEl]) {
        if self.meter.room(self.data, els.len()) {
            self.data.extend_from_slice(els);
        }
    }
}

struct StrokeCtx<'a> {
    // As a possible future optimization, we might not need separate storage
    // for forward and backward paths, we can add forward to the output in-place.
    // However, this structure is clearer and the cost fairly modest.
    output: Path<'a>,
    forward_path: Path<'a>,
    backward_path: Path<'a>,
    result_path: Path<'a>,
    start_pt: Point,
    start_norm: Vec2,
    start_tan: Vec2,
    last_pt: Point,
    last_tan: Vec2,
    // Precomputation of the join threshold, to optimize per-join logic.
    // If hypot < (hypot + dot) * join_thresh, omit join altogether.
    join_thresh: f64,
}

fn stroke_undashed(
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    tolerance: f64,
    ctx: &mut StrokeCtx<'_>,
) {
    for el in path {
        if ctx.output.meter.failed() {
            return;
        }
        let p0 = ctx.last_pt;
        match el {
            PathEl::MoveTo(p) => {
                ctx.finish(style);
                ctx.start_pt = p;
                ctx.last_pt = p;
            }
            PathEl::LineTo(p1) => {
                if p1 != p0 {
                    let tangent = p1 - p0;
                    ctx.do_join(style, tangent);
                    ctx.last_tan = tangent;
                    ctx.do_line(style, tangent, p1);
                }
            }
            PathEl::QuadTo(p1, p2) => {
                if p1 != p0 || p2 != p0 {
                    let q = QuadBez::new(p0, p1, p2);
                    let (tan0, tan1) = tangents(PathSeg::Quad(q));
                    ctx.do_join(style, tan0);
                    ctx.do_cubic(style, q.raise(), tolerance);
                    ctx.last_tan = tan1;
                }
            }
            PathEl::CurveTo(p1, p2, p3) => {
                if p1 != p0 || p2 != p0 || p3 != p0 {
                    let c = CubicBez::new(p0, p1, p2, p3);
                    let (tan0, tan1) = tangents(PathSeg::Cubic(c));
                    ctx.do_join(style, tan0);
                    ctx.do_cubic(style, c, tolerance);
                    ctx.last_tan = tan1;
                }
            }
            PathEl::ClosePath => {
                if p0 != ctx.start_pt {
                    let tangent = ctx.start_pt - p0;
                    ctx.do_join(style, tangent);
                    ctx.last_tan = tangent;
                    ctx.do_line(style, tangent, ctx.start_pt);
                }
                ctx.finish_closed(style);
            }
        }
    }
    ctx.finish(style);
}

fn round_cap(out: &mut Path<'_>, tolerance: f64, center: Point, norm: Vec2) {
    round_join(out, tolerance, center, norm, PI);
}

fn round_join(out: &mut Path<'_>, tolerance: f64, center: Point, norm: Vec2, angle: f64) {
    let a = Affine::new([norm.x, norm.y, -norm.y, norm.x, center.x, center.y]);
    let arc = Arc::new(Point::ORIGIN, (1.0, 1.0), PI - angle, angle, 0.0);
    arc.to_cubic_beziers(tolerance, |p1, p2, p3| out.curve_to(a * p1, a * p2, a * p3));
}

fn round_join_rev(out: &mut Path<'_>, tolerance: f64, center: Point, norm: Vec2, angle: f64) {
    let a = Affine::new([norm.x, norm.y, norm.y, -norm.x, center.x, center.y]);
    let arc = Arc::new(Point::ORIGIN, (1.0, 1.0), PI - angle, angle, 0.0);
    arc.to_cubic_beziers(tolerance, |p1, p2, p3| out.curve_to(a * p1, a * p2, a * p3));
}

fn square_cap(out: &mut Path<'_>, close: bool, center: Point, norm: Vec2) {
    let a = Affine::new([norm.x, norm.y, -norm.y, norm.x, center.x, center.y]);
    out.line_to(a * Point::new(1.0, 1.0));
    out.line_to(a * Point::new(-1.0, 1.0));
    if close {
        out.close_path();
    } else {
        out.line_to(a * Point::new(-1.0, 0.0));
    }
}

fn extend_reversed(out: &mut Path<'_>, elements: &[PathEl]) {
    for i in (1..elements.len()).rev() {
        if out.meter.failed() {
            return;
        }
        let end = elements[i - 1].end_point().unwrap();
        match elements[i] {
            PathEl::LineTo(_) => out.line_to(end),
            PathEl::QuadTo(p1, _) => out.quad_to(p1, end),
            PathEl::CurveTo(p1, p2, _) => out.curve_to(p2, p1, end),
            _ => unreachable!(),
        }
    }
}

impl StrokeCtx<'_> {
    /// Append forward and backward paths to output.
    fn finish(&mut self, style: &Stroke) {
        // TODO: scale
        let tolerance = 1e-3;
        if self.output.meter.failed() || self.forward_path.data.is_empty() {
            return;
        }
        self.output.extend(self.forward_path.data);
        let back_els = self.backward_path.data.as_slice();
        let return_p = back_els[back_els.len() - 1].end_point().unwrap();
        let d = self.last_pt - return_p;
        match style.end_cap {
            Cap::Butt => self.output.line_to(return_p),
            Cap::Round => round_cap(&mut self.output, tolerance, self.last_pt, d),
            Cap::Square => square_cap(&mut self.output, false, self.last_pt, d),
        }
        if self.output.meter.failed() {
            return;
        }
        extend_reversed(&mut self.output, back_els);
        match style.start_cap {
            Cap::Butt => self.output.close_path(),
            Cap::Round => round_cap(&mut self.output, tolerance, self.start_pt, self.start_norm),
            Cap::Square => square_cap(&mut self.output, true, self.start_pt, self.start_norm),
        }

        self.forward_path.data.clear();
        self.backward_path.data.clear();
    }

    /// Finish a closed path
    fn finish_closed(&mut self, style: &Stroke) {
        if self.output.meter.failed() || self.forward_path.data.is_empty() {
            return;
        }
        self.do_join(style, self.start_tan);
        if self.output.meter.failed() {
            return;
        }
        self.output.extend(self.forward_path.data);
        self.output.close_path();
        let back_els = self.backward_path.data.as_slice();
        let last_pt = back_els[back_els.len() - 1].end_point().unwrap();
        self.output.move_to(last_pt);
        if self.output.meter.failed() {
            return;
        }
        extend_reversed(&mut self.output, back_els);
        self.output.close_path();
        self.forward_path.data.clear();
        self.backward_path.data.clear();
    }

    fn do_join(&mut self, style: &Stroke, tan0: Vec2) {
        // TODO: scale
        let tolerance = 1e-3;
        let scale = 0.5 * style.width / tan0.hypot();
        let norm = scale * Vec2::new(-tan0.y, tan0.x);
        let p0 = self.last_pt;
        if self.forward_path.data.as_slice().is_empty() {
            self.forward_path.move_to(p0 - norm);
            self.backward_path.move_to(p0 + norm);
            self.start_tan = tan0;
            self.start_norm = norm;
        } else {
            let ab = self.last_tan;
            let cd = tan0;
            let cross = ab.cross(cd);
            let dot = ab.dot(cd);
            let hypot = cross.hypot(dot);
            // possible TODO: a minor speedup could be squaring both sides
            if dot <= 0.0 || cross.abs() >= hypot * self.join_thresh {
                match style.join {
                    Join::Bevel => {
                        self.forward_path.line_to(p0 - norm);
                        self.backward_path.line_to(p0 + norm);
                    }
                    Join::Miter => {
                        if 2.0 * hypot < (hypot + dot) * style.miter_limit.powi(2) {
                            // TODO: maybe better to store last_norm or derive from path?
                            let last_scale = 0.5 * style.width / ab.hypot();
                            let last_norm = last_scale * Vec2::new(-ab.y, ab.x);
                            if cross > 0.0 {
                                let fp_last = p0 - last_norm;
                                let fp_this = p0 - norm;
                                let h = ab.cross(fp_this - fp_last) / cross;
                                let miter_pt = fp_this - cd * h;
                                self.forward_path.line_to(miter_pt);
                                self.backward_path.line_to(p0);
                            } else if cross < 0.0 {
                                let fp_last = p0 + last_norm;
                                let fp_this = p0 + norm;
                                let h = ab.cross(fp_this - fp_last) / cross;
                                let miter_pt = fp_this - cd * h;
                                self.backward_path.line_to(miter_pt);
                                self.forward_path.line_to(p0);
                            }
                        }
                        self.forward_path.line_to(p0 - norm);
                        self.backward_path.line_to(p0 + norm);
                    }
                    Join::Round => {
                        let angle = cross.atan2(dot);
                        if angle > 0.0 {
                            self.backward_path.line_to(p0 + norm);
                            round_join(&mut self.forward_path, tolerance, p0, norm, angle);
                        } else {
                            self.forward_path.line_to(p0 - norm);
                            round_join_rev(&mut self.backward_path, tolerance, p0, -norm, -angle);
                        }
                    }
                }
            }
        }
    }

    fn do_line(&mut self, style: &Stroke, tangent: Vec2, p1: Point) {
        let scale = 0.5 * style.width / tangent.hypot();
        let norm = scale * Vec2::new(-tangent.y, tangent.x);
        self.forward_path.line_to(p1 - norm);
        self.backward_path.line_to(p1 + norm);
        self.last_pt = p1;
    }

    fn do_cubic(&mut self, style: &Stroke, c: CubicBez, tolerance: f64) {
        // First, detect degenerate linear case

        // Ordinarily, this is the direction of the chord, but if the chord is very
        // short, we take the longer control arm.
        let chord = c.p3 - c.p0;
        let mut chord_ref = chord;
        let mut chord_ref_hypot2 = chord_ref.hypot2();
        let d01 = c.p1 - c.p0;
        if d01.hypot2() > chord_ref_hypot2 {
            chord_ref = d01;
            chord_ref_hypot2 = chord_ref.hypot2();
        }
        let d23 = c.p3 - c.p2;
        if d23.hypot2() > chord_ref_hypot2 {
            chord_ref = d23;
            chord_ref_hypot2 = chord_ref.hypot2();
        }
        // Project Bézier onto chord
        let p0 = c.p0.to_vec2().dot(chord_ref);
        let p1 = c.p1.to_vec2().dot(chord_ref);
        let p2 = c.p2.to_vec2().dot(chord_ref);
        let p3 = c.p3.to_vec2().dot(chord_ref);
        const ENDPOINT_D: f64 = 0.01;
        if p3 <= p0
            || p1 > p2
            || p1 < p0 + ENDPOINT_D * (p3 - p0)
            || p2 > p3 - ENDPOINT_D * (p3 - p0)
        {
            // potentially a cusp inside
            let x01 = d01.cross(chord_ref);
            let x23 = d23.cross(chord_ref);
            let x03 = chord.cross(chord_ref);
            let thresh = tolerance.powi(2) * chord_ref_hypot2;
            if x01 * x01 < thresh && x23 * x23 < thresh && x03 * x03 < thresh {
                // control points are nearly co-linear
                let midpoint = c.p0.midpoint(c.p3);
                // Mapping back from projection of reference chord
                let ref_vec = chord_ref / chord_ref_hypot2;
                let ref_pt = midpoint - 0.5 * (p0 + p3) * ref_vec;
                self.do_linear(style, c, [p0, p1, p2, p3], ref_pt, ref_vec);
                return;
            }
        }

        offset::offset_cubic(c, -0.5 * style.width, tolerance, &mut self.result_path);
        self.forward_path
            .extend(self.result_path.data.get(1..).unwrap_or_default());
        offset::offset_cubic(c, 0.5 * style.width, tolerance, &mut self.result_path);
        self.backward_path
            .extend(self.result_path.data.get(1..).unwrap_or_default());
        self.last_pt = c.p3;
    }

    /// Do a cubic which is actually linear.
    ///
    /// The `p` argument is the control points projected to the reference chord.
    /// The ref arguments are the inverse map of a projection back to the client
    /// coordinate space.
    fn do_linear(
        &mut self,
        style: &Stroke,
        c: CubicBez,
        p: [f64; 4],
        ref_pt: Point,
        ref_vec: Vec2,
    ) {
        // Always do round join, to model cusp as limit of finite curvature (see Nehab).
        let style = Stroke::new(style.width).with_join(Join::Round);
        // Tangents of endpoints (for connecting to joins)
        let (tan0, tan1) = tangents(PathSeg::Cubic(c));
        self.last_tan = tan0;
        // find cusps
        let c0 = p[1] - p[0];
        let c1 = 2.0 * p[2] - 4.0 * p[1] + 2.0 * p[0];
        let c2 = p[3] - 3.0 * p[2] + 3.0 * p[1] - p[0];
        let roots = solve_quadratic(c0, c1, c2);
        // discard cusps right at endpoints
        const EPSILON: f64 = 1e-6;
        for t in roots {
            if t > EPSILON && t < 1.0 - EPSILON {
                let mt = 1.0 - t;
                let z = mt * (mt * mt * p[0] + 3.0 * t * (mt * p[1] + t * p[2])) + t * t * t * p[3];
                let p = ref_pt + z * ref_vec;
                let tan = p - self.last_pt;
                self.do_join(&style, tan);
                self.do_line(&style, tan, p);
                self.last_tan = tan;
            }
        }
        let tan = c.p3 - self.last_pt;
        self.do_join(&style, tan);
        self.do_line(&style, tan, c.p3);
        self.last_tan = tan;
        self.do_join(&style, tan1);
    }
}

fn tangents(seg: PathSeg) -> (Vec2, Vec2) {
    const EPS: f64 = 1e-12;
    match seg {
        PathSeg::Line(l) => {
            let d = l.p1 - l.p0;
            (d, d)
        }
        PathSeg::Quad(q) => {
            let d01 = q.p1 - q.p0;
            let d0 = if d01.hypot2() > EPS { d01 } else { q.p2 - q.p0 };
            let d12 = q.p2 - q.p1;
            let d1 = if d12.hypot2() > EPS { d12 } else { q.p2 - q.p0 };
            (d0, d1)
        }
        PathSeg::Cubic(c) => {
            let d01 = c.p1 - c.p0;
            let d0 = if d01.hypot2() > EPS {
                d01
            } else {
                let d02 = c.p2 - c.p0;
                if d02.hypot2() > EPS { d02 } else { c.p3 - c.p0 }
            };
            let d23 = c.p3 - c.p2;
            let d1 = if d23.hypot2() > EPS {
                d23
            } else {
                let d13 = c.p3 - c.p1;
                if d13.hypot2() > EPS { d13 } else { c.p3 - c.p0 }
            };
            (d0, d1)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kurbo::{self, BezPath, StrokeOpts},
        path_budget::PathBudget,
    };

    fn path() -> BezPath {
        let mut p = BezPath::new();
        p.move_to((2., 1.));
        p.line_to((8., 19.));
        p.quad_to((15., 3.), (20., 9.));
        p.curve_to((30., 11.), (3., 40.), (16., 22.));
        p.close_path();
        p.move_to((22., 2.));
        p.curve_to((30., 2.), (11., 2.), (19., 2.));
        p
    }

    #[test]
    fn outlines_match_kurbo() {
        let p = path();
        for cap in [Cap::Butt, Cap::Square, Cap::Round] {
            for join in [Join::Bevel, Join::Miter, Join::Round] {
                let solid = Stroke::new(5.25).with_caps(cap).with_join(join);
                let mut expected = kurbo::StrokeCtx::default();
                let mut actual = StrokeBuffers::default();
                let allowance = PathBudget::new(usize::MAX);
                for dashed in [false, true] {
                    let style = if dashed {
                        solid.clone().with_dashes(0.3, [2., 1.])
                    } else {
                        solid.clone()
                    };
                    kurbo::stroke_with(&p, &style, &StrokeOpts::default(), 0.25, &mut expected);
                    let meter = Meter::Raster(&allowance);
                    if dashed {
                        actual.expand(
                            kurbo::dash(p.iter(), style.dash_offset, &style.dash_pattern),
                            &solid,
                            0.25,
                            meter,
                        );
                    } else {
                        actual.expand(&p, &solid, 0.25, meter);
                    }
                    assert_eq!(actual.output(), expected.output().elements());
                }
            }
        }
    }

    #[test]
    fn refused_growth_leaves_no_outline_and_buffers_can_be_reused() {
        let p = path();
        let style = Stroke::new(8.);
        let mut complete = StrokeBuffers::default();
        let allowance = PathBudget::new(1 << 20);
        complete.expand(&p, &style, 0.25, Meter::Raster(&allowance));
        let used = (1 << 20) - allowance.remaining();
        let cap = |s: &StrokeBuffers| {
            [
                s.output.capacity(),
                s.forward.capacity(),
                s.backward.capacity(),
                s.result.capacity(),
            ]
            .iter()
            .sum::<usize>()
                * size_of::<PathEl>()
        };
        assert_eq!(used, cap(&complete));
        for limit in (0..used).step_by(size_of::<PathEl>()) {
            let budget = PathBudget::new(limit);
            let mut scratch = StrokeBuffers::default();
            scratch.expand(&p, &style, 0.25, Meter::Raster(&budget));
            assert!(cap(&scratch) <= limit);
            if Meter::Raster(&budget).failed() {
                assert!(scratch.output().is_empty());
            }
            scratch.expand(&p, &style, 0.25, Meter::default());
            assert_eq!(scratch.output(), complete.output());
        }
    }
}
