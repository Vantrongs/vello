// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A segment with huge coordinates costs time and memory bounded by the view, filled
//! or stroked, thin or wide, dashed or not.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use vello_common::fearless_simd::Level;
use vello_common::flatten::{FlattenCtx, Line, fill, stroke};
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath, Stroke, StrokeCtx};

/// Counts the bytes in use and their peak, and refuses requests over `CAP`, so code
/// that sizes a buffer by the coordinates aborts instead of exhausting the machine.
struct Capped;

const CAP: usize = 1 << 30;
static IN_USE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to `System`, only counting.
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > CAP {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract is `System`'s.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = IN_USE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        // SAFETY: `p` came from `alloc` above.
        unsafe { System.dealloc(p, layout) };
        IN_USE.fetch_sub(layout.size(), Relaxed);
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

const VIEW: RectU16 = RectU16 {
    x0: 0,
    y0: 0,
    x1: 100,
    y1: 100,
};

enum Draw {
    Fill,
    Stroke(Stroke),
}

/// Flattens `path` and returns its lines, asserting it stays under 64 MiB.
fn run(name: &str, path: &BezPath, draw: &Draw) -> Vec<Line> {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut lines = Vec::new();
    let (mut flatten_ctx, mut stroke_ctx) = (FlattenCtx::default(), StrokeCtx::default());
    let base = IN_USE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = Instant::now();
    match draw {
        Draw::Fill => fill(
            level,
            path.iter(),
            Affine::IDENTITY,
            &mut lines,
            &mut flatten_ctx,
            VIEW,
        ),
        Draw::Stroke(style) => stroke(
            level,
            path.iter(),
            style,
            Affine::IDENTITY,
            &mut lines,
            &mut flatten_ctx,
            &mut stroke_ctx,
            VIEW,
        ),
    }
    let peak = PEAK.load(Relaxed) - base;
    println!(
        "{name}: {} lines, peak {} KiB, {:?}",
        lines.len(),
        peak / 1024,
        t.elapsed()
    );
    assert!(peak < 64 << 20, "{name}: peak {peak} bytes");
    lines
}

/// Whether some line's bounding box overlaps the view.
fn reaches_view(lines: &[Line]) -> bool {
    lines.iter().any(|l| {
        let (x0, x1) = (l.p0.x.min(l.p1.x), l.p0.x.max(l.p1.x));
        let (y0, y1) = (l.p0.y.min(l.p1.y), l.p0.y.max(l.p1.y));
        x1 >= 0.0 && x0 <= 100.0 && y1 >= 0.0 && y0 <= 100.0
    })
}

/// Tests measure one at a time: the counters are global.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The review's cubic: collinear, control points at ±1e24, through the view.
fn collinear() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((10.0, 10.0));
    p.curve_to((1e24, 10.0), (-1e24, 10.0), (20.0, 10.0));
    p
}

fn bent() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((10.0, 10.0));
    p.curve_to((1e24, 10.0), (-1e24, 50.0), (20.0, 90.0));
    p.close_path();
    p
}

/// Vello's fill flattener emits lines for a quadratic in proportion to the square
/// root of its size.
fn quad() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((10.0, 10.0));
    p.quad_to((1e28, -1e28), (90.0, 90.0));
    p.close_path();
    p
}

fn long_line() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((-1e24, 50.0));
    p.line_to((1e24, 50.0));
    p
}

fn thin() -> Stroke {
    Stroke::new(1.0)
}

fn wide() -> Stroke {
    Stroke::new(3.0)
}

fn dashed(s: Stroke) -> Stroke {
    s.with_dashes(0.0, [2.0, 2.0])
}

fn check(name: &str, path: BezPath, draw: Draw) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lines = run(name, &path, &draw);
    assert!(reaches_view(&lines), "{name}: nothing in view");
}

#[test]
fn thin_collinear() {
    check("thin collinear", collinear(), Draw::Stroke(thin()));
}

#[test]
fn wide_collinear() {
    check("wide collinear", collinear(), Draw::Stroke(wide()));
}

#[test]
fn thin_bent() {
    check("thin bent", bent(), Draw::Stroke(thin()));
}

#[test]
fn wide_bent() {
    check("wide bent", bent(), Draw::Stroke(wide()));
}

#[test]
fn filled_bent() {
    check("filled bent", bent(), Draw::Fill);
}

#[test]
fn filled_quad() {
    check("filled quad", quad(), Draw::Fill);
}

#[test]
fn thin_quad() {
    check("thin quad", quad(), Draw::Stroke(thin()));
}

#[test]
fn dashed_thin_collinear() {
    check(
        "dashed thin collinear",
        collinear(),
        Draw::Stroke(dashed(thin())),
    );
}

#[test]
fn dashed_wide_collinear() {
    check(
        "dashed wide collinear",
        collinear(),
        Draw::Stroke(dashed(wide())),
    );
}

#[test]
fn dashed_thin_bent() {
    check("dashed thin bent", bent(), Draw::Stroke(dashed(thin())));
}

#[test]
fn dashed_thin_line() {
    check(
        "dashed thin line",
        long_line(),
        Draw::Stroke(dashed(thin())),
    );
}

#[test]
fn dashed_wide_line() {
    check(
        "dashed wide line",
        long_line(),
        Draw::Stroke(dashed(wide())),
    );
}
