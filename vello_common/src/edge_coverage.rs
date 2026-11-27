// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Edge queries for intersections of prepared fill and clip paths.

#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Affine, BezPath, PathEl, Stroke};
use crate::kurbo::{Line, ParamCurveNearest, Point, Rect, Vec2};
use crate::path_budget::CapacityExceeded;
use crate::peniko::Fill;
use crate::strip_generator::{PathStyle, PreparedPath, StripGenerator};
use crate::{geometry::RectU16, path_budget::PathBudget};
use alloc::vec::Vec;
use fearless_simd::Level;

/// Admission of work units and temporary storage before gathering or testing edges.
pub trait EdgeBudget {
    /// Capacity available to prepare the next path.
    fn remaining_bytes(&self) -> usize;
    /// Reserve work and storage, returning false if the query should stop.
    fn admit(&mut self, work: u64, bytes: usize) -> bool;
}

/// Intersection of paths indexed by whole device rows.
#[derive(Debug)]
pub struct EdgeCoverage {
    first_row: i64,
    paths: Vec<(Fill, Vec<Vec<Line>>)>,
}

impl EdgeCoverage {
    /// Start an intersection whose first sampled device row is `first_row`.
    fn new(first_row: i64) -> Self {
        Self {
            first_row,
            paths: Vec::new(),
        }
    }
    /// Include one more path in the intersection.
    fn intersect(
        &mut self,
        rule: Fill,
        rows: EdgeRows,
        meter: &mut dyn EdgeBudget,
    ) -> Result<(), CapacityExceeded> {
        assert_eq!(rows.first, self.first_row);
        assert!(
            self.paths
                .first()
                .is_none_or(|(_, r)| r.len() == rows.lines.len())
        );
        reserve(&mut self.paths, 1, meter)?;
        self.paths.push((rule, rows.lines));
        Ok(())
    }
    /// Whether any paths were supplied.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
    /// How often the region's one path winds around `p`, which lies in its rows.
    pub fn winding(&self, p: Point) -> i32 {
        let row = (p.y.floor() as i64 - self.first_row) as usize;
        winding(self.paths[0].1[row].iter(), p)
    }

    /// The lines of the device row `y`, of all the paths.
    fn lines(&self, y: i64) -> usize {
        let row = (y - self.first_row) as usize;
        self.paths.iter().map(|(_, rows)| rows[row].len()).sum()
    }

    /// Whether every path holds `p`, which lies in the region's rows: it winds
    /// around `p` as its rule fills. A test looks at the lines of `p`'s row once.
    fn holds(&self, p: Point) -> bool {
        let row = (p.y.floor() as i64 - self.first_row) as usize;
        self.paths
            .iter()
            .all(|(rule, rows)| fills(*rule, winding(rows[row].iter(), p)))
    }

    /// A point of the pixel at `x`, `y` the region holds, for a pixel none of
    /// whose subsamples it holds: the middle of the widest stretch of the
    /// region along the pixel's rows and columns of subsamples, `SUBSAMPLES`
    /// each, which a part of it any taller or wider than the steps between them
    /// crosses. `None` where none does: what the region holds of the pixel then
    /// lies within one of those steps each way, too little to tell in its
    /// colour. Each line of the pixel's row is looked at once along each, and
    /// the crossings sorted; the work is taken of `meter` first.
    fn point_within(
        &self,
        x: i64,
        y: i64,
        meter: &mut dyn EdgeBudget,
    ) -> Result<Option<Point>, CapacityExceeded> {
        let row = (y - self.first_row) as usize;
        let lines = self.lines(y) as u64;
        // A look at each line, and sorting its crossing, along each row and column.
        let along = lines * (2 + u64::from(lines.max(1).ilog2()));
        if !meter.admit(along.saturating_mul(2 * SUBSAMPLES as u64), 0) {
            return Err(CapacityExceeded);
        }
        let mut crossings: Vec<(f64, usize, i32)> = Vec::new();
        reserve(&mut crossings, lines as usize, meter)?;
        let mut windings = Vec::new();
        reserve(&mut windings, self.paths.len(), meter)?;
        let mut widest: Option<(f64, Point)> = None;
        let n = SUBSAMPLES as f64;
        for k in 0..SUBSAMPLES {
            let at = (k as f64 + 0.5) / n;
            for across in [true, false] {
                // From `start`, one pixel along `step`.
                let (start, step) = if across {
                    (Point::new(x as f64, y as f64 + at), Vec2::new(1.0, 0.0))
                } else {
                    (Point::new(x as f64 + at, y as f64), Vec2::new(0.0, 1.0))
                };
                crossings.clear();
                for (i, (_, rows)) in self.paths.iter().enumerate() {
                    let lines = &rows[row];

                    for line in lines.iter() {
                        if let Some((t, change)) = crossing(line, start, across) {
                            crossings.push((t, i, change));
                        }
                    }
                }
                crossings.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                // The pixel boundary may itself be an edge. Initialize in the
                // first open interval, whose right endpoint is the first crossing.
                let first = start + step * (0.5 * crossings.first().map_or(1.0, |c| c.0));
                let mut inside = 0;
                windings.clear();
                for (rule, rows) in &self.paths {
                    let w = winding(rows[row].iter(), first);
                    inside += usize::from(fills(*rule, w));
                    windings.push(w);
                }
                let mut from = 0.0;
                for &(t, i, change) in crossings.iter().chain([&(1.0, usize::MAX, 0)]) {
                    if inside == self.paths.len() && t - from > widest.map_or(0.0, |w| w.0) {
                        let p = start + step * ((from + t) / 2.0);
                        if !meter.admit(lines * 2, 0) {
                            return Err(CapacityExceeded);
                        }
                        if p.y < y as f64 + 1.0
                            && p.x < x as f64 + 1.0
                            && self.holds(p)
                            && self.paths.iter().all(|(_, rows)| {
                                rows[row].iter().all(|line| {
                                    let a = line.p0 - p;
                                    let b = line.p1 - p;
                                    a.cross(b) != 0.0 || a.dot(b) > 0.0
                                })
                            })
                        {
                            widest = Some((t - from, p));
                        }
                    }
                    if i == usize::MAX {
                        break;
                    }
                    let rule = self.paths[i].0;
                    inside -= usize::from(fills(rule, windings[i]));
                    windings[i] += change;
                    inside += usize::from(fills(rule, windings[i]));
                    from = t;
                }
            }
        }
        Ok(widest.map(|(_, p)| p))
    }

    /// Whether a line of a path lies nearer `p`, which lies in the region's
    /// rows, than `distance`.
    fn near(&self, p: Point, distance: f64) -> bool {
        let row = (p.y.floor() as i64 - self.first_row) as usize;
        self.paths.iter().any(|(_, rows)| {
            rows[row]
                .iter()
                .any(|line| line.nearest(p, 0.0).distance_sq < distance * distance)
        })
    }

    /// Whether a line of a path passes through the device pixel at `x`, `y`.
    fn crosses(&self, x: i64, y: i64) -> bool {
        let row = (y - self.first_row) as usize;
        let pixel = Rect::new(x as f64, y as f64, (x + 1) as f64, (y + 1) as f64);
        self.paths.iter().any(|(_, rows)| {
            rows[row]
                .iter()
                .any(|line| clip_line(*line, pixel).is_some())
        })
    }
}

/// How often `lines` wind around `p`: a line counts where the line along the
/// device x axis from `p` to the left crosses it, going up or down, its lower
/// end its own and its upper one not. The canvas counts from the left too, and
/// leaves out what lies right of the pixels it is culled to.
fn winding<'a>(lines: impl Iterator<Item = &'a Line>, p: Point) -> i32 {
    lines
        .filter(|line| (line.p0.y <= p.y) != (line.p1.y <= p.y))
        .filter(|line| {
            let (a, b) = (line.p0, line.p1);
            a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) < p.x
        })
        .map(|line| if line.p1.y > line.p0.y { -1 } else { 1 })
        .sum()
}

/// Whether `rule` fills where its path winds `winding` times.
fn fills(rule: Fill, winding: i32) -> bool {
    match rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    }
}

/// Where `line` crosses the pixel-long stretch from `start` along the device x
/// axis (`across`) or y axis, as a share of it within (0, 1], and how the winding
/// changes past it (`winding`): the crossings of the line along x the winding
/// counts, so along x the winding gains the line's count as it passes it, and
/// along y gains a line going right and loses one going left.
fn crossing(line: &Line, start: Point, across: bool) -> Option<(f64, i32)> {
    let (a, b) = (line.p0, line.p1);
    if across {
        if (a.y <= start.y) == (b.y <= start.y) {
            return None;
        }
        let x = a.x + (start.y - a.y) * (b.x - a.x) / (b.y - a.y);
        let t = x - start.x;
        (t > 0.0 && t <= 1.0).then_some((t, if b.y > a.y { -1 } else { 1 }))
    } else {
        if (a.x <= start.x) == (b.x <= start.x) {
            return None;
        }
        let y = a.y + (start.x - a.x) * (b.y - a.y) / (b.x - a.x);
        let t = y - start.y;
        (t > 0.0 && t <= 1.0).then_some((t, if b.x > a.x { 1 } else { -1 }))
    }
}

/// The part of `line` within `rect`, `None` when it misses it.
fn clip_line(line: Line, rect: Rect) -> Option<Line> {
    let (p, d) = (line.p0, line.p1 - line.p0);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (q, r) in [
        (-d.x, p.x - rect.x0),
        (d.x, rect.x1 - p.x),
        (-d.y, p.y - rect.y0),
        (d.y, rect.y1 - p.y),
    ] {
        if q == 0.0 {
            if r < 0.0 {
                return None;
            }
        } else {
            let t = r / q;
            if q < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    (t0 < t1).then(|| Line::new(p + d * t0, p + d * t1))
}

/// One path's lines by the device rows of the pixels tested they reach, each line
/// placed and its bytes held of the meter before it is kept. A line wholly right
/// of `right` is not kept: a point tested lies left of it, and a line counts
/// towards its winding only to its left.
#[derive(Debug)]
struct EdgeRows {
    first: i64,
    right: f64,
    lines: Vec<Vec<Line>>,
}

impl EdgeRows {
    fn new(pixels: Rect, right: f64, meter: &mut dyn EdgeBudget) -> Result<Self, CapacityExceeded> {
        let height = pixels.height().max(0.0) as usize;
        let mut lines = Vec::new();
        reserve(&mut lines, height, meter)?;
        if !meter.admit(height as u64, 0) {
            return Err(CapacityExceeded);
        }
        lines.resize_with(height, Vec::new);
        Ok(Self {
            first: pixels.y0 as i64,
            right,
            lines,
        })
    }

    /// Keeps `line` in each row it reaches.
    fn keep(&mut self, line: Line, meter: &mut dyn EdgeBudget) -> Result<(), CapacityExceeded> {
        if line.p0.x.min(line.p1.x) > self.right {
            return Ok(());
        }
        let (y0, y1) = (line.p0.y.min(line.p1.y), line.p0.y.max(line.p1.y));
        if self.lines.is_empty() {
            return Ok(());
        }
        let last = self.first.saturating_add(self.lines.len() as i64 - 1);
        let (from, to) = (
            (y0.floor() as i64).max(self.first),
            (y1.floor() as i64).min(last),
        );
        if from > to {
            return Ok(());
        }
        let places = (to - from + 1) as usize;
        if !meter.admit(places as u64, 0) {
            return Err(CapacityExceeded);
        }
        for row in &mut self.lines[(from - self.first) as usize..=(to - self.first) as usize] {
            reserve(row, 1, meter)?;
            row.push(line);
        }
        Ok(())
    }
}

fn reserve<T>(
    vec: &mut Vec<T>,
    extra: usize,
    meter: &mut dyn EdgeBudget,
) -> Result<(), CapacityExceeded> {
    if extra <= vec.capacity() - vec.len() {
        return Ok(());
    }
    let cap = vec
        .len()
        .checked_add(extra)
        .ok_or(CapacityExceeded)?
        .max(vec.capacity().saturating_mul(2))
        .max(4);
    let bytes = (cap - vec.capacity())
        .checked_mul(size_of::<T>())
        .ok_or(CapacityExceeded)?;
    if cap.saturating_mul(size_of::<T>()) > meter.remaining_bytes() || !meter.admit(0, bytes) {
        return Err(CapacityExceeded);
    }
    vec.try_reserve_exact(cap - vec.len())
        .map_err(|_| CapacityExceeded)
}

/// The `k`th of the `SUBSAMPLES` points a pixel at a step is averaged from, the
/// pixel's corner at `x`, `y`: one in each row and each column of an n × n
/// grid, the rows taken in strides of `SUBSAMPLE_STRIDE`, so an edge along
/// either axis is resolved in n steps, one at a slant as well.
pub fn subsample(x: f64, y: f64, k: usize) -> Point {
    let n = SUBSAMPLES as f64;
    let row = (k * SUBSAMPLE_STRIDE) % SUBSAMPLES;
    Point::new(x + (k as f64 + 0.5) / n, y + (row as f64 + 0.5) / n)
}

/// Number of coverage samples per pixel, one in each row and column of a square grid.
pub const SUBSAMPLES: usize = 16;

/// The rows between the points of consecutive columns (`SUBSAMPLES`); prime
/// to it, so each row has one, and as far from 1 and its half as that allows,
/// so the points spread over the pixel rather than along a diagonal.
const SUBSAMPLE_STRIDE: usize = 5;

impl PreparedPath<'_, '_> {
    /// Index this path's directed edges for batched pixel queries, translated by
    /// `origin` into the coordinate system of the whole-pixel rectangle `pixels`.
    /// Use [`EdgeCoverage::prepare`] with the original clip paths when their
    /// subpixel intersection is needed; alpha strips retain only pixel coverage.
    pub fn edge_coverage(
        &self,
        pixels: Rect,
        origin: Vec2,
        meter: &mut dyn EdgeBudget,
    ) -> Result<EdgeCoverage, CapacityExceeded> {
        let mut rows = EdgeRows::new(pixels, pixels.x1 + MARGIN, meter)?;
        self.index_edges(origin, &mut rows, meter)?;
        let mut coverage = EdgeCoverage::new(pixels.y0 as i64);
        coverage.intersect(self.fill_rule(), rows, meter)?;
        Ok(coverage)
    }

    /// Index the directed edges of this subpath at their device-space origin.
    fn index_edges(
        &self,
        origin: Vec2,
        rows: &mut EdgeRows,
        meter: &mut dyn EdgeBudget,
    ) -> Result<(), CapacityExceeded> {
        if !meter.admit(self.edges().len() as u64, 0) {
            return Err(CapacityExceeded);
        }
        let point = |p: crate::flatten::Point| Point::new(p.x.into(), p.y.into()) + origin;
        for edge in self.edges() {
            rows.keep(Line::new(point(edge.p0), point(edge.p1)), meter)?;
        }
        Ok(())
    }
}

/// Coverage of a requested pixel cut by an edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeSamples {
    /// Bits indexing covered points in the subsample sequence.
    Mask(u16),
    /// A point strictly inside an intersection missed by the regular samples.
    Interior(Point),
}

impl EdgeCoverage {
    /// Visit partial sample masks and sliver points for the requested device pixels in one indexed row.
    pub fn sample_row(
        &self,
        y: i64,
        xs: impl IntoIterator<Item = i64>,
        tolerance: f64,
        meter: &mut dyn EdgeBudget,
        mut visit: impl FnMut(i64, EdgeSamples),
    ) -> Result<(), CapacityExceeded> {
        let lines = self.lines(y) as u64;
        if lines == 0 {
            return Ok(());
        }
        for x in xs {
            if !meter.admit(lines, 0) {
                return Err(CapacityExceeded);
            }
            if !self.crosses(x, y) {
                continue;
            }
            if !meter.admit(2 * lines * SUBSAMPLES as u64, 0) {
                return Err(CapacityExceeded);
            }
            let mut held = 0u16;
            for k in 0..SUBSAMPLES {
                let p = subsample(x as f64, y as f64, k);
                if self.holds(p) && !self.near(p, tolerance) {
                    held |= 1 << k;
                }
            }
            if held == 0 {
                if let Some(p) = self.point_within(x, y, meter)? {
                    visit(x, EdgeSamples::Interior(p));
                }
            } else if held != u16::MAX {
                visit(x, EdgeSamples::Mask(held));
            }
        }
        Ok(())
    }
}
/// The shape a paint covers, placed in device space by its transform: a path
/// filled by its rule, or stroked in its style, as the canvas draws it.
#[derive(Debug, Clone, Copy)]
pub enum Painted<'a> {
    /// Path filled with the given winding rule.
    Fill(&'a BezPath, Affine, Fill),
    /// Stroke expanded in path space before transformation.
    Stroke(&'a BezPath, Affine, &'a Stroke),
}

/// Device-space padding around the sampling window for nearby edges.
const MARGIN: f64 = 1.0;

/// The steps the canvas's code takes for an element of a path, besides the lines
/// it makes: placing it, testing it against the pixels it is culled to, and for a
/// stroke, its outline's sides, join and cap. A segment far larger than the view is
/// split first, in steps bounded by the view (`vello_common` `cull.rs`).
const ELEMENT_COST: u64 = 16;

impl EdgeCoverage {
    /// The region `paints` cover together, tested in the device pixels `pixels`,
    /// whole ones: their lines as the canvas, drawing at `level`, makes them,
    /// culled to `pixels` and `MARGIN` around them, whose work is taken of `meter`
    /// before each path is given to the canvas's code and whose bytes before they
    /// are kept. That code's own buffers for a path are those the canvas fills for
    /// it, whose cull is the whole view.
    pub fn prepare<'a>(
        paints: impl IntoIterator<Item = Painted<'a>>,
        pixels: Rect,
        level: Level,
        meter: &mut dyn EdgeBudget,
    ) -> Result<EdgeCoverage, CapacityExceeded> {
        let window = pixels.inflate(MARGIN, MARGIN);
        let mut canvas = CanvasLines::new(level, window);
        let mut paths = EdgeCoverage::new(pixels.y0 as i64);
        for paint in paints {
            let mut rows = EdgeRows::new(pixels, window.x1, meter)?;
            let rule = match paint {
                Painted::Fill(path, to_device, rule) => {
                    canvas.pay(path, meter)?;
                    canvas.prepare(
                        path.elements(),
                        to_device,
                        PathStyle::Fill(rule),
                        &mut rows,
                        meter,
                    )?;
                    rule
                }
                Painted::Stroke(path, to_device, style) => {
                    canvas.pay(path, meter)?;
                    stroke(path, to_device, style, &mut canvas, &mut rows, meter)?;
                    Fill::NonZero
                }
            };
            paths.intersect(rule, rows, meter)?;
        }
        Ok(paths)
    }
}

/// Reuses preparation buffers across paths in the same sampling window.
struct CanvasLines {
    window: Rect,
    generator: StripGenerator,
    origin: Vec2,
}

impl CanvasLines {
    fn new(level: Level, window: Rect) -> Self {
        let bbox = device_box(window);
        Self {
            window,
            origin: Vec2::new(bbox.x0.into(), bbox.y0.into()),
            generator: StripGenerator::new(bbox.x1 - bbox.x0, bbox.y1 - bbox.y0, level),
        }
    }

    /// Takes of `meter` the work of giving `path`'s elements to the canvas's
    /// code.
    fn pay(&self, path: &BezPath, meter: &mut dyn EdgeBudget) -> Result<(), CapacityExceeded> {
        let elements = path.elements().len();

        if meter.admit(ELEMENT_COST.saturating_mul(elements as u64), 0) {
            Ok(())
        } else {
            Err(CapacityExceeded)
        }
    }

    fn prepare(
        &mut self,
        elements: &[PathEl],
        to_device: Affine,
        style: PathStyle<'_>,
        rows: &mut EdgeRows,
        meter: &mut dyn EdgeBudget,
    ) -> Result<(), CapacityExceeded> {
        let available = meter.remaining_bytes();
        let capacity = PathBudget::new(available);
        let prepared = self.generator.prepare_path(
            elements.iter().copied(),
            style,
            Affine::translate(-self.origin) * to_device,
            None,
            &capacity,
        )?;
        if !meter.admit(0, available - capacity.remaining()) {
            return Err(CapacityExceeded);
        }
        prepared.index_edges(self.origin, rows, meter)
    }
}

fn stroke(
    path: &BezPath,
    to_device: Affine,
    style: &Stroke,
    canvas: &mut CanvasLines,
    rows: &mut EdgeRows,
    meter: &mut dyn EdgeBudget,
) -> Result<(), CapacityExceeded> {
    let elements = path.elements();
    if path.is_nan() {
        return Ok(());
    }
    let window = canvas.window;
    let view = window.width().max(window.height());
    let reach = crate::flatten::stroke_reach(style, crate::flatten::max_scale(to_device), true);
    let reached = window.inflate(reach, reach);
    let mut start = 0;
    while start < elements.len() {
        let end = elements[start + 1..]
            .iter()
            .position(|el| matches!(el, PathEl::MoveTo(_)))
            .map_or(elements.len(), |i| start + 1 + i);
        let subpath = &elements[start..end];
        start = end;
        if misses(control_box(subpath, to_device), reached) {
            continue;
        }
        let walk = crate::path_budget::dash_work(&style.dash_pattern, subpath, to_device, view);
        if !meter.admit(walk, 0) {
            return Err(CapacityExceeded);
        }
        canvas.prepare(subpath, to_device, PathStyle::Stroke(style), rows, meter)?;
    }
    Ok(())
}

/// The device box of the points of `elements` placed by `to_device`, which holds
/// them.
fn control_box(elements: &[PathEl], to_device: Affine) -> Rect {
    let mut bounds = Rect::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for el in elements {
        each_point(el, |p| bounds = bounds.union_pt(to_device * p));
    }
    bounds
}

/// Passes on the points of `el`.
fn each_point(el: &PathEl, mut f: impl FnMut(Point)) {
    match *el {
        PathEl::MoveTo(p) | PathEl::LineTo(p) => f(p),
        PathEl::QuadTo(p1, p2) => [p1, p2].into_iter().for_each(f),
        PathEl::CurveTo(p1, p2, p3) => [p1, p2, p3].into_iter().for_each(f),
        PathEl::ClosePath => {}
    }
}

/// Whether `bounds` lies wholly outside `window`.
fn misses(bounds: Rect, window: Rect) -> bool {
    bounds.x1 < window.x0 || bounds.x0 > window.x1 || bounds.y1 < window.y0 || bounds.y0 > window.y1
}

/// The device pixels holding `rect`, as the canvas's code culls to them.
fn device_box(rect: Rect) -> RectU16 {
    let clamp = |v: f64| v.clamp(0.0, f64::from(u16::MAX)) as u16;
    RectU16::new(
        clamp(rect.x0.floor()),
        clamp(rect.y0.floor()),
        clamp(rect.x1.ceil()),
        clamp(rect.y1.ceil()),
    )
}

#[cfg(test)]
mod edge_tests {
    use super::*;
    use crate::flatten;
    struct Unlimited;
    impl EdgeBudget for Unlimited {
        fn remaining_bytes(&self) -> usize {
            usize::MAX
        }
        fn admit(&mut self, _: u64, _: usize) -> bool {
            true
        }
    }

    /// Prepared edges may be clipped at the window while the unmetered SIMD
    /// flattener keeps their offscreen endpoints. Both must enclose the same
    /// samples across open, closed, curved and transformed paths.
    #[test]
    fn prepared_edges_preserve_winding_inside_the_window() {
        let level = Level::new();
        let pixels = Rect::new(0.0, 0.0, 120.0, 90.0);
        let window = pixels.inflate(MARGIN, MARGIN);
        let mut paths: Vec<BezPath> = Vec::new();
        let mut p = BezPath::new();
        p.move_to((10.0, 10.0));
        p.line_to((100.0, 15.0));
        p.curve_to((130.0, 40.0), (-20.0, 60.0), (60.0, 85.0));
        p.quad_to((0.0, 100.0), (5.0, 40.0));
        p.close_path();
        p.curve_to((40.0, 0.0), (80.0, 90.0), (110.0, 30.0));
        p.move_to((50.0, 50.0));
        p.move_to((20.0, 70.0));
        p.line_to((70.0, 20.0));
        p.line_to((90.0, 80.0));
        p.move_to((30.0, 30.0));
        p.line_to((30.0, 30.0));
        p.close_path();
        paths.push(p);
        let mut culled = BezPath::new();
        culled.move_to((60.0, 45.0));
        culled.curve_to((500.0, 10.0), (600.0, 80.0), (700.0, 45.0));
        culled.curve_to((60.0, -400.0), (90.0, -500.0), (60.0, -300.0));
        culled.curve_to((60.0, 400.0), (90.0, 500.0), (60.0, 300.0));
        culled.curve_to((-400.0, 10.0), (-500.0, 80.0), (-300.0, 45.0));
        culled.line_to((60.0, 45.0));
        paths.push(culled);
        let mut huge = BezPath::new();
        huge.move_to((10.0, 10.0));
        huge.curve_to((1e7, 20.0), (-1e7, 70.0), (100.0, 80.0));
        huge.line_to((-3e5, 40.0));
        huge.line_to((3e5, 45.0));
        huge.curve_to((5000.0, -5000.0), (-5000.0, 5000.0), (90.0, 20.0));
        huge.close_path();
        paths.push(huge);
        let mut far = BezPath::new();
        far.move_to((10.0, 10.0));
        far.curve_to((1e15, 1e15), (-1e15, 1e15), (20.0, 60.0));
        far.line_to((90.0, 30.0));
        paths.push(far);
        let mut cubics = BezPath::new();
        for k in 0..64 {
            let y = f64::from(k);
            cubics.move_to((0.0, y));
            cubics.curve_to((65536.0, y), (65536.0, 32768.0 + y), (0.0, 98304.0 + y));
            cubics.close_path();
        }
        paths.push(cubics);
        let transforms = [
            Affine::IDENTITY,
            Affine::rotate(0.3).then_translate(Vec2::new(30.0, -10.0)),
            Affine::new([1.5, 0.2, -0.7, 0.9, 10.0, 5.0]),
            Affine::scale(1e-3),
        ];
        for (i, path) in paths.iter().enumerate() {
            for to_device in transforms {
                let mut canvas = CanvasLines::new(level, window);
                let mut meter = Unlimited;
                let available = meter.remaining_bytes();
                let capacity = PathBudget::new(available);
                let prepared = canvas
                    .generator
                    .prepare_path(
                        path,
                        PathStyle::Fill(Fill::NonZero),
                        to_device,
                        None,
                        &capacity,
                    )
                    .unwrap();
                let coverage = prepared
                    .edge_coverage(pixels, canvas.origin, &mut meter)
                    .unwrap();
                let mut lines = Vec::new();
                flatten::fill(
                    level,
                    path.iter(),
                    to_device,
                    &mut lines,
                    &mut flatten::FlattenCtx::default(),
                    device_box(window),
                );
                // The reference edges are independent segments, so count each left-ray crossing.
                for row in 0..pixels.height() as usize {
                    for x in 0..pixels.width() as usize {
                        for (dx, dy) in [(0.125, 0.375), (0.625, 0.875)] {
                            let point = Point::new(x as f64 + dx, row as f64 + dy);
                            assert_eq!(
                                coverage.winding(point),
                                lines
                                    .iter()
                                    .map(|line| {
                                        let (x0, y0, x1, y1) = (
                                            line.p0.x as f64,
                                            line.p0.y as f64,
                                            line.p1.x as f64,
                                            line.p1.y as f64,
                                        );
                                        if (y0 <= point.y) != (y1 <= point.y)
                                            && x0 + (point.y - y0) * (x1 - x0) / (y1 - y0) < point.x
                                        {
                                            if y1 > y0 { -1 } else { 1 }
                                        } else {
                                            0
                                        }
                                    })
                                    .sum::<i32>(),
                                "path {i}, {to_device:?}, {point:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
