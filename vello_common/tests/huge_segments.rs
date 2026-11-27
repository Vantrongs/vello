// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A segment with huge coordinates costs time and memory bounded by the view, filled
//! or stroked, thin or wide, dashed or not, under any transform: one that compresses
//! a direction (so the path is far larger than its device image), a nearly singular
//! one, and a tiny scale that makes dashes far finer than a pixel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use vello_common::fearless_simd::Level;
use vello_common::flatten::{FlattenCtx, Line, fill, stroke};
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath, Join, Stroke, StrokeCtx};

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

/// Flattens `path` under `affine` and returns its lines, asserting it stays under
/// 64 MiB.
fn run(name: &str, path: &BezPath, draw: &Draw, affine: Affine) -> Vec<Line> {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut lines = Vec::new();
    let (mut flatten_ctx, mut stroke_ctx) = (FlattenCtx::default(), StrokeCtx::default());
    let base = IN_USE.load(Relaxed);
    PEAK.store(base, Relaxed);
    match draw {
        Draw::Fill => fill(
            level,
            path.iter(),
            affine,
            &mut lines,
            &mut flatten_ctx,
            VIEW,
        ),
        Draw::Stroke(style) => stroke(
            level,
            path.iter(),
            style,
            affine,
            &mut lines,
            &mut flatten_ctx,
            &mut stroke_ctx,
            VIEW,
        ),
    }
    let peak = PEAK.load(Relaxed).saturating_sub(base);
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

/// A collinear cubic with control points at ±1e24, through the view.
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
    check_under(name, path, draw, Affine::IDENTITY);
}

fn check_under(name: &str, path: BezPath, draw: Draw, affine: Affine) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lines = run(name, &path, &draw, affine);
    assert!(reaches_view(&lines), "{name}: nothing in view");
}

/// Compresses x by 1e20: the `collinear` cubic, 2e24 wide in path space, is 2e4 px wide
/// on the device, through the view along y = 10.
fn squash() -> Affine {
    Affine::scale_non_uniform(1e-20, 1.0)
}

/// A nearly singular transform (determinant 1e-20): x compressed by 1e20 and sheared
/// into y, rotated by 30°, with (10, 10) at (50, 50).
fn nearly_singular() -> Affine {
    let linear =
        Affine::rotate(30_f64.to_radians()) * Affine::new([1e-20, 0.0, 0.3, 1.0, 0.0, 0.0]);
    let at = linear * vello_common::kurbo::Point::new(10.0, 10.0);
    Affine::translate((50.0 - at.x, 50.0 - at.y)) * linear
}

/// A line 1e6 long, 100 px under a scale of 1e-4, across the view along y = 50.
fn tiny_line() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((0.0, 5e5));
    p.line_to((1e6, 5e5));
    p
}

/// Dashes of 0.01: 5e7 of them on `tiny_line`, 5e5 per device pixel.
fn fine_dashes(s: Stroke) -> Stroke {
    s.with_dashes(0.0, [0.01, 0.01])
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

/// The winding number of the polygons `lines` around `p`, counted from the left:
/// the fill drops lines right of the view, which change no winding in it.
fn winding(lines: &[Line], p: (f32, f32)) -> i32 {
    let mut w = 0;
    for l in lines {
        let (a, b) = (l.p0, l.p1);
        if (a.y <= p.1) != (b.y <= p.1) {
            let x = a.x + (p.1 - a.y) / (b.y - a.y) * (b.x - a.x);
            if x < p.0 {
                w += if b.y > a.y { 1 } else { -1 };
            }
        }
    }
    w
}

/// Under `squash`, the `collinear` cubic is the line y = 10 through the view: a stroke
/// `half` wide each side covers it and nothing 2 px away.
fn squashed_band(name: &str, style: Stroke, half: f32) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lines = run(name, &collinear(), &Draw::Stroke(style), squash());
    for x in 0..100 {
        let x = x as f32 + 0.5;
        assert_ne!(winding(&lines, (x, 10.0)), 0, "{name}: ({x}, 10) uncovered");
        for y in [10.0 - half + 0.05, 10.0 + half - 0.05] {
            assert_ne!(winding(&lines, (x, y)), 0, "{name}: ({x}, {y}) uncovered");
        }
        for y in [10.0 - half - 2.0, 10.0 + half + 2.0] {
            assert_eq!(winding(&lines, (x, y)), 0, "{name}: ({x}, {y}) covered");
        }
    }
}

#[test]
fn squashed_thin_collinear() {
    squashed_band("squashed thin collinear", thin(), 0.5);
}

#[test]
fn squashed_wide_collinear() {
    squashed_band("squashed wide collinear", wide(), 1.5);
}

#[test]
fn squashed_thin_bent() {
    check_under("squashed thin bent", bent(), Draw::Stroke(thin()), squash());
}

#[test]
fn squashed_filled_bent() {
    check_under("squashed filled bent", bent(), Draw::Fill, squash());
}

#[test]
fn squashed_thin_quad() {
    check_under("squashed thin quad", quad(), Draw::Stroke(thin()), squash());
}

#[test]
fn squashed_dashed_thin_collinear() {
    check_under(
        "squashed dashed thin collinear",
        collinear(),
        Draw::Stroke(dashed(thin())),
        squash(),
    );
}

#[test]
fn squashed_dashed_wide_collinear() {
    check_under(
        "squashed dashed wide collinear",
        collinear(),
        Draw::Stroke(dashed(wide())),
        squash(),
    );
}

#[test]
fn nearly_singular_thin() {
    check_under(
        "nearly singular thin",
        bent(),
        Draw::Stroke(Stroke::new(0.5)),
        nearly_singular(),
    );
}

#[test]
fn nearly_singular_dashed_thin() {
    check_under(
        "nearly singular dashed thin",
        bent(),
        Draw::Stroke(dashed(thin())),
        nearly_singular(),
    );
}

#[test]
fn nearly_singular_wide() {
    check_under(
        "nearly singular wide",
        bent(),
        Draw::Stroke(wide()),
        nearly_singular(),
    );
}

#[test]
fn nearly_singular_filled() {
    check_under(
        "nearly singular filled",
        bent(),
        Draw::Fill,
        nearly_singular(),
    );
}

#[test]
fn fine_dashes_thin() {
    check_under(
        "fine dashes thin",
        tiny_line(),
        Draw::Stroke(fine_dashes(Stroke::new(1e4))),
        Affine::scale(1e-4),
    );
}

#[test]
fn fine_dashes_wide() {
    check_under(
        "fine dashes wide",
        tiny_line(),
        Draw::Stroke(fine_dashes(Stroke::new(3e4))),
        Affine::scale(1e-4),
    );
}

/// A miter limit that widens the stroke's view so far that the first piece in it
/// spends the segment's polyline lines; the pieces after it must not reach kurbo's
/// flattener as curves of their path-space size.
fn mitred(width: f64, limit: f64) -> Stroke {
    Stroke::new(width)
        .with_join(Join::Miter)
        .with_miter_limit(limit)
}

/// Under `squash`, the cubic (10, 10) C (1e10, 10)
/// (-1e10, 10) (20, 10) on the device.
fn squashed_far() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((1e21, 10.0));
    p.curve_to((1e30, 10.0), (-1e30, 10.0), (2e21, 10.0));
    p
}

/// A cubic swinging out to 1e24, untransformed.
fn swung() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((10.0, 10.0));
    p.curve_to((1e24, 0.0), (-1e24, 1e24), (20.0, 10.0));
    p
}

#[test]
fn spent_polyline_squashed() {
    check_under(
        "spent polyline squashed",
        squashed_far(),
        Draw::Stroke(mitred(1.0, 2e9)),
        squash(),
    );
}

#[test]
fn spent_polyline_squashed_dashed() {
    check_under(
        "spent polyline squashed dashed",
        squashed_far(),
        Draw::Stroke(dashed(mitred(1.0, 2e9))),
        squash(),
    );
}

#[test]
fn spent_polyline_swung() {
    check(
        "spent polyline swung",
        swung(),
        Draw::Stroke(mitred(0.5, 1.1e24)),
    );
}

#[test]
fn spent_polyline_swung_dashed() {
    check(
        "spent polyline swung dashed",
        swung(),
        Draw::Stroke(dashed(mitred(0.5, 1.1e24))),
    );
}

/// A line 64 980 px long across the view along y = 50.
fn long_device_line() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((-32_440.0, 50.0));
    p.line_to((32_540.0, 50.0));
    p
}

/// 10 000 entries of 1e-5, a period of 0.1: on `long_device_line` only 649 800
/// periods, but 6.5e9 dashes and gaps, as many as the two-entry `[1e-5, 1e-5]` has.
fn many_entries(s: Stroke, entry: f64) -> Stroke {
    s.with_dashes(0.0, vec![entry; 10_000])
}

/// A pattern of many tiny entries costs what the equivalent two-entry pattern does,
/// and covers the same (where their subpaths break differs: the phase at a piece's
/// end rounds differently).
fn many_entries_as_two(name: &str, s: Stroke, half: f32) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let path = long_device_line();
    let two = s.clone().with_dashes(0.0, [1e-5, 1e-5]);
    let a = run(
        name,
        &path,
        &Draw::Stroke(many_entries(s, 1e-5)),
        Affine::IDENTITY,
    );
    let b = run(name, &path, &Draw::Stroke(two), Affine::IDENTITY);
    assert!(reaches_view(&a), "{name}: nothing in view");
    for x in 0..200 {
        for y in [45.0, 50.05 - half, 50.0, 49.95 + half, 55.0] {
            let p = (x as f32 * 0.5 + 0.25, y);
            let covered = |l: &[Line]| winding(l, p) != 0;
            assert_eq!(covered(&a), covered(&b), "{name}: {p:?}");
            assert_eq!(
                covered(&a),
                (50.0 - half..=50.0 + half).contains(&y),
                "{name}: {p:?}"
            );
        }
    }
}

#[test]
fn many_entries_thin() {
    many_entries_as_two("many entries thin", thin(), 0.5);
}

#[test]
fn many_entries_wide() {
    many_entries_as_two("many entries wide", wide(), 1.5);
}

/// Entries of 1 px on the `collinear` cubic: few in view, but each piece skipped on
/// the way down from 1e24 walks up to a whole period, 10 000 entries.
#[test]
fn many_entries_on_hidden_pieces() {
    check(
        "many entries on hidden pieces",
        collinear(),
        Draw::Stroke(many_entries(wide(), 1.0)),
    );
}

/// A shape whose far parts are at a given magnitude.
type Scaled = (&'static str, fn(f64) -> BezPath);

/// A fill's work follows what is visible, not its coordinates: the same visible
/// geometry with its far parts at 1e6 or at 1e300 flattens to about as many lines,
/// in about as much memory.
#[test]
fn fill_work_does_not_grow_with_magnitude() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let shapes: [Scaled; 4] = [
        ("bent", |m| {
            let mut p = BezPath::new();
            p.move_to((10.0, 10.0));
            p.curve_to((m, 10.0), (-m, 50.0), (20.0, 90.0));
            p.close_path();
            p
        }),
        ("slanted", |m| {
            let mut p = BezPath::new();
            p.move_to((10.0, 10.0));
            p.curve_to((m, m), (-m, m), (90.0, 20.0));
            p.close_path();
            p
        }),
        ("quad", |m| {
            let mut p = BezPath::new();
            p.move_to((10.0, 10.0));
            p.quad_to((m, -m), (90.0, 90.0));
            p.close_path();
            p
        }),
        ("triangle", |m| {
            let mut p = BezPath::new();
            p.move_to((-m, -m + 0.3));
            p.line_to((m, m + 0.3));
            p.line_to((-m, m));
            p.close_path();
            p
        }),
    ];
    for (name, shape) in shapes {
        let mut counts = Vec::new();
        for m in [1e6, 1e12, 1e24, 1e48, 1e100, 1e200, 1e300] {
            let lines = run(
                &format!("{name} {m:e}"),
                &shape(m),
                &Draw::Fill,
                Affine::IDENTITY,
            );
            counts.push(lines.len());
        }
        let most = counts.iter().copied().max().unwrap_or(0);
        assert!(most <= counts[0] + 8, "{name}: lines {counts:?}");
    }
}

#[test]
fn hidden_prefix_preparation_has_a_fixed_allocation_bound() {
    use vello_common::path_budget::PathBudget;
    use vello_common::strip_generator::{PathStyle, StripGenerator, StripStorage};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let style = Stroke::new(2.)
        .with_caps(vello_common::kurbo::Cap::Butt)
        .with_dashes(0., [7., 3.]);
    for exp in [12, 41, 82, 203, 900] {
        let mut path = BezPath::new();
        path.move_to((-2.0_f64.powi(exp), 20.));
        path.line_to((100., 20.));
        let mut generator = StripGenerator::new(100, 100, Level::baseline());
        let base = IN_USE.load(Relaxed);
        PEAK.store(base, Relaxed);
        let prepared = generator
            .prepare_path(
                path.iter(),
                PathStyle::Stroke(&style),
                Affine::IDENTITY,
                None,
                &PathBudget::new(16384),
            )
            .unwrap();
        assert!(PEAK.load(Relaxed).saturating_sub(base) < 65536, "2^{exp}");
        assert!(!prepared.edges().is_empty());
        assert!(prepared.edges().len() < 128, "2^{exp}");
        let phase = [6, 2, 4, 8][(exp % 4) as usize];
        for x in 0..100 {
            assert_eq!(
                winding(prepared.edges(), (x as f32 + 0.5, 19.5)) != 0,
                (x + phase) % 10 < 7
            );
        }
        let mut strips = StripStorage::default();
        prepared.submit(None, &mut strips);
        for (y, row) in coverage::<100, 100>(&strips).iter().enumerate() {
            for (x, alpha) in row.iter().enumerate() {
                let on = (19..21).contains(&y) && (x + phase) % 10 < 7;
                assert_eq!(*alpha, if on { 255 } else { 0 }, "2^{exp} pixel ({x},{y})");
            }
        }
    }
}

#[test]
fn rejected_preparation_can_retry_without_partial_edges() {
    use vello_common::path_budget::{CapacityExceeded, PathBudget};
    use vello_common::strip_generator::{PathStyle, StripGenerator};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let path = bent();
    let style = PathStyle::Fill(peniko::Fill::NonZero);
    let mut generator = StripGenerator::new(100, 100, Level::baseline());
    assert_eq!(
        generator
            .prepare_path(
                path.iter(),
                style,
                Affine::IDENTITY,
                None,
                &PathBudget::new(0)
            )
            .unwrap_err(),
        CapacityExceeded
    );
    let retried = generator
        .prepare_path(
            path.iter(),
            style,
            Affine::IDENTITY,
            None,
            &PathBudget::new(65536),
        )
        .unwrap();
    let mut fresh = StripGenerator::new(100, 100, Level::baseline());
    let reference = fresh
        .prepare_path(
            path.iter(),
            style,
            Affine::IDENTITY,
            None,
            &PathBudget::new(65536),
        )
        .unwrap();
    assert!(!retried.edges().is_empty());
    assert!(retried.edges().iter().all(|l| {
        [l.p0, l.p1]
            .iter()
            .all(|p| (0.0..=100.).contains(&p.x) && (0.0..=100.).contains(&p.y))
    }));
    assert!(
        retried
            .edges()
            .iter()
            .map(|l| (l.p0, l.p1))
            .eq(reference.edges().iter().map(|l| (l.p0, l.p1)))
    );
}

#[test]
fn submitting_inspected_edges_preserves_the_hole() {
    use vello_common::kurbo::{Rect, Shape};
    use vello_common::path_budget::PathBudget;
    use vello_common::strip_generator::{PathStyle, StripGenerator, StripStorage};

    let hole = Rect::new(3., 2., 7., 6.);
    let mut path = Rect::new(-1e30, -1e30, 1e30, 1e30).to_path(0.1);
    path.extend(hole.to_path(0.1).reverse_subpaths().iter());
    let mut generator = StripGenerator::new(12, 8, Level::baseline());
    let prepared = generator
        .prepare_path(
            path.iter(),
            PathStyle::Fill(peniko::Fill::NonZero),
            Affine::IDENTITY,
            None,
            &PathBudget::new(1024),
        )
        .unwrap();
    for y in 0..8 {
        for x in 0..12 {
            let inside = winding(prepared.edges(), (x as f32 + 0.5, y as f32 + 0.5)) != 0;
            assert_eq!(inside, !((3..7).contains(&x) && (2..6).contains(&y)));
        }
    }
    let mut storage = StripStorage::default();
    prepared.submit(None, &mut storage);
    let pixels = coverage::<12, 8>(&storage);
    for (y, row) in pixels.iter().enumerate() {
        for (x, alpha) in row.iter().enumerate() {
            let hole = (3..7).contains(&x) && (2..6).contains(&y);
            assert_eq!(*alpha, if hole { 0 } else { 255 });
        }
    }
}

fn coverage<const W: usize, const H: usize>(
    storage: &vello_common::strip_generator::StripStorage,
) -> [[u8; W]; H] {
    use vello_common::strip::visit_strip_fill_segments;
    let mut pixels = [[0; W]; H];
    let viewport = RectU16::new(0, 0, W.try_into().unwrap(), H.try_into().unwrap());
    visit_strip_fill_segments(
        &storage.strips,
        viewport.covering_tiles(),
        &mut pixels,
        |pixels, seg| {
            let x0 = usize::from(seg.x0());
            for (dy, row) in pixels
                .iter_mut()
                .skip(usize::from(seg.y()))
                .take(4)
                .enumerate()
            {
                for (dx, alpha) in row[x0..(seg.x1() as usize).min(W)].iter_mut().enumerate() {
                    *alpha = storage.alphas[seg.alpha_column as usize * 4 + dx * 4 + dy];
                }
            }
        },
        |pixels, seg| {
            for row in pixels.iter_mut().skip(usize::from(seg.y())).take(4) {
                row[usize::from(seg.x0())..(seg.x1() as usize).min(W)].fill(255);
            }
        },
    );
    pixels
}

#[test]
fn finite_curves_preserve_visible_coverage_and_allow_retry() {
    use vello_common::path_budget::{CapacityExceeded, PathBudget};
    use vello_common::strip_generator::{PathStyle, StripGenerator, StripStorage};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut right = BezPath::new();
    right.move_to((10., 10.));
    right.curve_to((5e307, 10.), (10., 30.), (20., 30.));
    let mut swung = BezPath::new();
    swung.move_to((4., -3.2e-306));
    swung.curve_to((4e307, 32. / 3.), (-4e307, 64. / 3.), (4., 32.));
    let mut quad = BezPath::new();
    quad.move_to((-1e308, 0.));
    quad.quad_to((1.5e308, 8.), (-1e308, 16.));
    for (mut path, on, off) in [
        (right, (24, 13), (5, 13)),
        (swung, (24, 8), (24, 24)),
        (quad, (24, 8), (24, 0)),
    ] {
        path.close_path();
        let mut genr = StripGenerator::new(64, 64, Level::baseline());
        let style = PathStyle::Fill(peniko::Fill::NonZero);
        assert_eq!(
            genr.prepare_path(
                path.iter(),
                style,
                Affine::IDENTITY,
                None,
                &PathBudget::new(0)
            )
            .unwrap_err(),
            CapacityExceeded
        );
        let prepared = genr
            .prepare_path(
                path.iter(),
                style,
                Affine::IDENTITY,
                None,
                &PathBudget::new(65536),
            )
            .unwrap();
        let mut strips = StripStorage::default();
        prepared.submit(None, &mut strips);
        let pixels = coverage::<64, 64>(&strips);
        assert_eq!(pixels[on.1][on.0], 255, "{on:?}");
        assert_eq!(pixels[off.1][off.0], 0, "{off:?}");
    }
}

#[test]
fn wide_dashed_stroke_budget_boundary() {
    use vello_common::{
        path_budget::PathBudget,
        strip_generator::{GenerationMode, StripGenerator, StripStorage},
    };
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut path = BezPath::new();
    path.move_to((5., 4.));
    path.curve_to((90., 4.), (3., 88.), (95., 90.));
    let style = Stroke::new(12.)
        .with_join(Join::Round)
        .with_dashes(1., [5., 3.]);
    let run = |bytes: usize| {
        let mut generator = StripGenerator::new(100, 100, Level::baseline());
        let mut storage = StripStorage::new(GenerationMode::Append);
        generator.generate_filled_path(
            &path,
            peniko::Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );
        let previous = (storage.strips.clone(), storage.alphas.clone());
        let base = IN_USE.load(Relaxed);
        PEAK.store(base, Relaxed);
        let accepted = generator
            .try_generate_stroke(
                &path,
                &style,
                Affine::IDENTITY,
                None,
                &mut storage,
                None,
                &PathBudget::new(bytes),
            )
            .is_ok();
        let peak = PEAK.load(Relaxed).saturating_sub(base);
        if accepted {
            assert!(storage.strips.len() > previous.0.len());
        } else {
            assert_eq!(storage.strips, previous.0);
            assert_eq!(storage.alphas, previous.1);
        }
        (accepted, peak)
    };
    let (mut lo, mut hi) = (0usize, 1 << 20);
    assert!(!run(lo).0 && run(hi).0);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if run(mid).0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let (ok, peak) = run(hi);
    let (refused, rejected_peak) = run(lo);
    assert!(ok && !refused);
    eprintln!(
        "wide dashed stroke accepted {hi} B (peak +{peak} B), refused {lo} B (peak +{rejected_peak} B)"
    );
}

#[test]
fn generated_curves_fit_the_element_line_bound() {
    use vello_common::kurbo::{PathEl, Point, Rect};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1_u64 << 53) as f64
    };
    let shaped = [
        [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 1.5)],
        [(0.0, 0.0), (2.0, 1.0), (-1.0, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.5, 0.0)],
        [(0.0, 0.0), (1.0, 0.001), (0.0, 0.002), (1.0, 0.003)],
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1e-6), (0.0, 0.0)],
    ];
    let mut curves = |size: f64, n: usize| {
        let mut all: Vec<[Point; 4]> = (0..n)
            .map(|_| {
                std::array::from_fn(|_| {
                    Point::new((random() - 0.3) * size, (random() - 0.3) * size)
                })
            })
            .collect();
        all.extend(shaped.map(|c| c.map(|(x, y)| Point::new(x * size, y * size))));
        all
    };
    let level = Level::new();

    for size in [10.0, 1e3, 1e4, 65536.0, 131_072.0, 1e6, 1e9, 1e20] {
        for [p0, p1, p2, p3] in curves(size, 400) {
            for width in [100.0, 4096.0, 65535.0] {
                let window = Rect::new(0.0, 0.0, width, width);
                for el in [PathEl::CurveTo(p1, p2, p3), PathEl::QuadTo(p1, p2)] {
                    let mut lines = Vec::new();
                    fill(
                        level,
                        [PathEl::MoveTo(p0), el],
                        Affine::IDENTITY,
                        &mut lines,
                        &mut FlattenCtx::default(),
                        RectU16::new(0, 0, window.x1 as u16, window.y1 as u16),
                    );
                    assert!(lines.len() <= 4096, "{p0:?} {el:?}: {}", lines.len());
                }
            }
        }
    }
}
