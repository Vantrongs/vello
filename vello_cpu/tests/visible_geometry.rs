// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fills whose geometry reaches far beyond the view, or whose winding numbers are
//! large, and dashes too dense to walk, render the pixels the geometry covers: checked
//! pixel by pixel against an independent reference, under an allocator that refuses
//! any request over 1 GiB.

use std::alloc::{GlobalAlloc, Layout, System};
use vello_cpu::kurbo::{
    Affine, BezPath, Cap, Circle, CubicBez, ParamCurve, ParamCurveExtrema, PathEl, PathSeg, Point,
    Rect, Shape, Stroke,
};
use vello_cpu::peniko::Fill;
use vello_cpu::{Pixmap, RenderContext, Resources, color::palette::css::BLACK};

/// Refuses requests over `CAP`, so code that sizes a buffer by the coordinates aborts
/// instead of exhausting the machine.
struct Capped;

const CAP: usize = 1 << 30;

// SAFETY: forwards to `System`.
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > CAP {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        // SAFETY: `p` came from `alloc` above.
        unsafe { System.dealloc(p, layout) };
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

const SIZE: u16 = 100;

/// The alpha of each pixel, row by row, of `path` filled black under `affine`.
fn render(path: &BezPath, rule: Fill, affine: Affine) -> Vec<u8> {
    draw(|ctx| {
        ctx.set_fill_rule(rule);
        ctx.set_transform(affine);
        ctx.fill_path(path);
    })
}

/// The alpha of each pixel, row by row, of what `paint` draws black.
fn draw(paint: impl FnOnce(&mut RenderContext)) -> Vec<u8> {
    let mut ctx = RenderContext::new(SIZE, SIZE);
    ctx.set_paint(BLACK);
    paint(&mut ctx);
    ctx.flush();
    let mut pixmap = Pixmap::new(SIZE, SIZE);
    ctx.render(&mut pixmap, &mut Resources::new());
    pixmap.data().iter().map(|p| p.a).collect()
}

/// The segments of `path` under `affine` (as the renderer transforms them, in `f64`),
/// each subpath closed.
fn device_segments(path: &BezPath, affine: Affine) -> Vec<PathSeg> {
    let mut segs = Vec::new();
    let (mut start, mut last) = (Point::ZERO, Point::ZERO);
    let close = |segs: &mut Vec<PathSeg>, last: Point, start: Point| {
        if last != start {
            segs.push(PathSeg::Line(vello_cpu::kurbo::Line::new(last, start)));
        }
    };
    for el in path.elements() {
        match affine * *el {
            PathEl::MoveTo(p) => {
                close(&mut segs, last, start);
                (start, last) = (p, p);
            }
            PathEl::LineTo(p) => {
                segs.push(PathSeg::Line(vello_cpu::kurbo::Line::new(last, p)));
                last = p;
            }
            PathEl::QuadTo(p1, p2) => {
                segs.push(PathSeg::Quad(vello_cpu::kurbo::QuadBez::new(last, p1, p2)));
                last = p2;
            }
            PathEl::CurveTo(p1, p2, p3) => {
                segs.push(PathSeg::Cubic(CubicBez::new(last, p1, p2, p3)));
                last = p3;
            }
            PathEl::ClosePath => {
                close(&mut segs, last, start);
                last = start;
            }
        }
    }
    close(&mut segs, last, start);
    segs
}

/// Where `seg` crosses the row y = `c`, and in which direction, found on the curve
/// itself: split at its extrema, then bisected to adjacent `f64` parameters.
fn crossings(seg: &PathSeg, c: f64, out: &mut Vec<(f64, i32)>) {
    if let PathSeg::Line(l) = *seg {
        let (a, b) = (l.p0, l.p1);
        if (a.y <= c) != (b.y <= c) {
            let x = if (c - a.y).abs() <= (c - b.y).abs() {
                a.x + (c - a.y) / (b.y - a.y) * (b.x - a.x)
            } else {
                b.x + (c - b.y) / (a.y - b.y) * (a.x - b.x)
            };
            out.push((x, if b.y > a.y { 1 } else { -1 }));
        }
        return;
    }
    let mut ts: Vec<f64> = seg.extrema().into_iter().collect();
    ts.sort_by(f64::total_cmp);
    ts.insert(0, 0.0);
    ts.push(1.0);
    let at = |t: f64| match t {
        0.0 => seg.start(),
        1.0 => seg.end(),
        _ => seg.eval(t),
    };
    for w in ts.windows(2) {
        let (mut lo, mut hi) = (w[0], w[1]);
        let (y0, y1) = (at(lo).y, at(hi).y);
        if (y0 <= c) == (y1 <= c) {
            continue;
        }
        loop {
            let mid = lo + 0.5 * (hi - lo);
            if !(mid > lo && mid < hi) {
                break;
            }
            if (seg.eval(mid).y <= c) == (y0 <= c) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Between adjacent parameters the curve is the chord joining them (which, for
        // coordinates around 1e30, can be 1e14 px long).
        let (a, b) = (at(lo), at(hi));
        let x = if (c - a.y).abs() <= (c - b.y).abs() {
            a.x + (c - a.y) / (b.y - a.y) * (b.x - a.x)
        } else {
            b.x + (c - b.y) / (a.y - b.y) * (a.x - b.x)
        };
        out.push((x, if y1 > y0 { 1 } else { -1 }));
    }
}

/// The coverage of each pixel by the fill of `segs`, from 16 × 16 samples each.
fn coverage(segs: &[PathSeg], rule: Fill) -> Vec<f64> {
    const N: usize = 16;
    let size = usize::from(SIZE);
    let mut cov = vec![0.0; size * size];
    let mut row = Vec::new();
    for sy in 0..size * N {
        let c = (sy as f64 + 0.5) / N as f64;
        row.clear();
        for seg in segs {
            crossings(seg, c, &mut row);
        }
        row.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut k = 0;
        let mut winding = 0;
        for sx in 0..size * N {
            let x = (sx as f64 + 0.5) / N as f64;
            while k < row.len() && row[k].0 < x {
                winding += row[k].1;
                k += 1;
            }
            let inside = match rule {
                Fill::NonZero => winding != 0,
                Fill::EvenOdd => winding % 2 != 0,
            };
            if inside {
                cov[(sy / N) * size + sx / N] += 1.0 / (N * N) as f64;
            }
        }
    }
    cov
}

/// Renders `path` filled under `affine` and compares each pixel with the sampled
/// coverage: within the flattening tolerance (a quarter pixel) plus the sampling's
/// error, and close on average.
fn matches_coverage(name: &str, path: &BezPath, rule: Fill, affine: Affine) {
    let alphas = render(path, rule, affine);
    let cov = coverage(&device_segments(path, affine), rule);
    let mut worst = (0.0, 0);
    let mut sum = 0.0;
    for (i, (&a, &c)) in alphas.iter().zip(&cov).enumerate() {
        let d = (f64::from(a) - 255.0 * c).abs();
        sum += d;
        if d > worst.0 {
            worst = (d, i);
        }
    }
    let (x, y) = (worst.1 % usize::from(SIZE), worst.1 / usize::from(SIZE));
    let mean = sum / alphas.len() as f64;
    println!("{name}: worst {:.1} at ({x}, {y}), mean {mean:.3}", worst.0);
    assert!(
        worst.0 <= 90.0,
        "{name}: pixel ({x}, {y}) has alpha {}, coverage {:.3}",
        alphas[worst.1],
        cov[worst.1]
    );
    assert!(mean <= 2.0, "{name}: mean difference {mean:.3}");
}

/// `n` copies of the square `r`, clockwise on the device (y down) or not.
fn squares(p: &mut BezPath, r: Rect, n: usize, clockwise: bool) {
    let corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)];
    for _ in 0..n {
        p.move_to(corners[0]);
        for i in 1..4 {
            p.line_to(corners[if clockwise { i } else { 4 - i }]);
        }
        p.close_path();
    }
}

/// Asserts every pixel has alpha `a` (opaque or transparent).
fn uniform(name: &str, alphas: &[u8], a: u8) {
    let wrong = alphas.iter().filter(|&&x| x != a).count();
    assert_eq!(wrong, 0, "{name}: {wrong} pixels are not {a}");
}

/// Many contours in one fill: the winding counter of rows left of the view (and of
/// tiles in it) holds any count, also a multiple of 2^16.
#[test]
fn winding_numbers_beyond_16_bits() {
    let around = Rect::new(-10.0, -10.0, 110.0, 110.0);
    let inside = Rect::new(10.0, 10.0, 90.0, 90.0);
    let cases: [(&str, Rect, usize, usize, Fill, u8); 6] = [
        (
            "65536 around the view",
            around,
            65_536,
            0,
            Fill::NonZero,
            255,
        ),
        (
            "131072 around less 65536",
            around,
            131_072,
            65_536,
            Fill::NonZero,
            255,
        ),
        (
            "65536 each way around",
            around,
            65_536,
            65_536,
            Fill::NonZero,
            0,
        ),
        ("65536 in view", inside, 65_536, 0, Fill::NonZero, 255),
        ("65536 even-odd", around, 65_536, 0, Fill::EvenOdd, 0),
        ("65537 even-odd", around, 65_537, 0, Fill::EvenOdd, 255),
    ];
    for (name, r, cw, ccw, rule, a) in cases {
        let mut p = BezPath::new();
        squares(&mut p, r, cw, true);
        squares(&mut p, r, ccw, false);
        let alphas = render(&p, rule, Affine::IDENTITY);
        if r == around {
            uniform(name, &alphas, a);
        } else {
            // The interior of the inner square, away from its anti-aliased edges.
            let interior: Vec<u8> = (0..usize::from(SIZE) * usize::from(SIZE))
                .filter(|i| {
                    let (x, y) = (i % 100, i / 100);
                    (12..88).contains(&x) && (12..88).contains(&y)
                })
                .map(|i| alphas[i])
                .collect();
            uniform(name, &interior, a);
        }
    }
}

fn cubic(p: [(f64, f64); 4]) -> BezPath {
    let mut path = BezPath::new();
    path.move_to(p[0]);
    path.curve_to(p[1], p[2], p[3]);
    path.close_path();
    path
}

/// Both rules, for shapes whose rules agree or whose holes are the test.
const RULES: [Fill; 2] = [Fill::NonZero, Fill::EvenOdd];

/// Cubics with control points far beyond `f32` (and `f64` squares) whose ends are in
/// the view: near its ends each runs straight to the view's edge, and the parts far
/// out still count in the view's winding.
#[test]
fn huge_controls_with_visible_ends() {
    let cases = [
        (
            "bent 1e24",
            [(10.0, 10.0), (1e24, 10.0), (-1e24, 50.0), (20.0, 90.0)],
        ),
        (
            "slanted 1e30",
            [(10.0, 10.0), (1e30, 1e30), (-1e30, 1e30), (90.0, 20.0)],
        ),
        (
            "slanted 1e300",
            [(10.0, 10.0), (1e300, 1e300), (-1e300, 1e300), (90.0, 20.0)],
        ),
        (
            "swung 1e24",
            [(10.0, 10.0), (1e24, 0.0), (-1e24, 1e24), (20.0, 90.0)],
        ),
    ];
    for (name, p) in cases {
        for rule in RULES {
            matches_coverage(
                &format!("{name} {rule:?}"),
                &cubic(p),
                rule,
                Affine::IDENTITY,
            );
        }
    }
    let mut quad = BezPath::new();
    quad.move_to((10.0, 10.0));
    quad.quad_to((1e28, -1e28), (90.0, 90.0));
    quad.close_path();
    matches_coverage("quad 1e28", &quad, Fill::NonZero, Affine::IDENTITY);
}

/// The fills of `tests/huge_segments.rs` under a transform that compresses x by 1e20,
/// and under a nearly singular one; with controls at 1e24 and at 1e300.
#[test]
fn huge_controls_under_degenerate_transforms() {
    let squash = Affine::scale_non_uniform(1e-20, 1.0);
    let linear =
        Affine::rotate(30_f64.to_radians()) * Affine::new([1e-20, 0.0, 0.3, 1.0, 0.0, 0.0]);
    let at = linear * Point::new(10.0, 10.0);
    let nearly_singular = Affine::translate((50.0 - at.x, 50.0 - at.y)) * linear;
    for m in [1e24, 1e300] {
        let bent = cubic([(10.0, 10.0), (m, 10.0), (-m, 50.0), (20.0, 90.0)]);
        for (name, affine) in [("squashed", squash), ("nearly singular", nearly_singular)] {
            matches_coverage(&format!("{name} {m:e}"), &bent, Fill::NonZero, affine);
        }
    }
}

/// An ellipse of four cubics, clockwise on the device, through the middles of the
/// sides of the box `left, top, right, bottom` and tangent to them there. The sides
/// are given exactly, and the middle `c` apart from them: at 1e300, `30 + r - r` is 0.
fn oval(left: f64, top: f64, right: f64, bottom: f64, c: Point) -> BezPath {
    const K: f64 = 0.552_284_749_830_793_4;
    let (kx, ky) = (K * (right - left) / 2.0, K * (bottom - top) / 2.0);
    let mut p = BezPath::new();
    p.move_to((right, c.y));
    p.curve_to((right, c.y + ky), (c.x + kx, bottom), (c.x, bottom));
    p.curve_to((c.x - kx, bottom), (left, c.y + ky), (left, c.y));
    p.curve_to((left, c.y - ky), (c.x - kx, top), (c.x, top));
    p.curve_to((c.x + kx, top), (right, c.y - ky), (right, c.y));
    p.close_path();
    p
}

/// A circle of radius `r` around the view's middle (as an oval: at 1e300 its middle
/// is lost in its sides).
fn around(r: f64) -> BezPath {
    oval(
        50.0 - r,
        50.0 - r,
        50.0 + r,
        50.0 + r,
        Point::new(50.0, 50.0),
    )
}

/// Shapes far larger than the view enclose it, with holes in it, by either rule.
#[test]
fn enclosing_fills_and_holes() {
    let hole = Circle::new((40.0, 60.0), 30.0).to_path(0.1);
    for r in [1e9, 1e30, 1e300] {
        let around = around(r);
        let mut ring = around.clone();
        ring.extend(hole.iter());
        let mut reversed = around.clone();
        reversed.extend(hole.reverse_subpaths().iter());
        let square = Rect::new(-r, -r, r, r).to_path(0.1);
        let mut square_ring = square.clone();
        square_ring.extend(hole.iter());
        for rule in RULES {
            for (name, path) in [
                ("around", &around),
                ("ring", &ring),
                ("reversed ring", &reversed),
                ("square ring", &square_ring),
            ] {
                matches_coverage(
                    &format!("{name} {r:e} {rule:?}"),
                    path,
                    rule,
                    Affine::IDENTITY,
                );
            }
        }
    }
}

/// Huge ovals touching the middle of each edge of the view from inside and from
/// outside: the clip finds tangent crossings without losing or doubling a piece.
#[test]
fn huge_curves_tangent_to_each_edge() {
    for r in [1e6, 1e300] {
        let d = 2.0 * r;
        let cases = [
            ("top, inside", oval(-r, 0.0, r, d, Point::new(50.0, r))),
            ("top, outside", oval(-r, -d, r, 0.0, Point::new(50.0, -r))),
            (
                "bottom, inside",
                oval(-r, 100.0 - d, r, 100.0, Point::new(50.0, 100.0 - r)),
            ),
            (
                "bottom, outside",
                oval(-r, 100.0, r, 100.0 + d, Point::new(50.0, 100.0 + r)),
            ),
            ("left, inside", oval(0.0, -r, d, r, Point::new(r, 50.0))),
            ("left, outside", oval(-d, -r, 0.0, r, Point::new(-r, 50.0))),
            (
                "right, inside",
                oval(100.0 - d, -r, 100.0, r, Point::new(100.0 - r, 50.0)),
            ),
            (
                "right, outside",
                oval(100.0, -r, 100.0 + d, r, Point::new(100.0 + r, 50.0)),
            ),
        ];
        for (name, path) in cases {
            matches_coverage(
                &format!("{name} {r:e}"),
                &path,
                Fill::NonZero,
                Affine::IDENTITY,
            );
        }
    }
}

/// Huge contours of opposite direction cancel in the view where both enclose it.
#[test]
fn opposite_windings_cancel() {
    for r in [1e9, 1e300] {
        let d = 2.0 * r;
        let a = oval(-r, 30.0, r, d, Point::new(50.0, r));
        let b = oval(-r, 70.0, r, d, Point::new(50.0, r));
        let mut both = a.clone();
        both.extend(a.reverse_subpaths().iter());
        matches_coverage(
            &format!("reversed {r:e}"),
            &both,
            Fill::NonZero,
            Affine::IDENTITY,
        );
        let mut band = a.clone();
        band.extend(b.reverse_subpaths().iter());
        for rule in RULES {
            matches_coverage(
                &format!("band {r:e} {rule:?}"),
                &band,
                rule,
                Affine::IDENTITY,
            );
        }
    }
}

/// Lines 2e9 px long through the view: as `f32` their position near the view was
/// lost (a 1 px step at 2e7 px), so a fill's diagonal edge moved by up to 32 px.
#[test]
fn long_lines_through_the_view() {
    let mut triangle = BezPath::new();
    triangle.move_to((-1e9, -1e9 + 0.3));
    triangle.line_to((1e9, 1e9 + 0.3));
    triangle.line_to((-1e9, 1e9));
    triangle.close_path();
    for rule in RULES {
        matches_coverage(
            &format!("diagonal {rule:?}"),
            &triangle,
            rule,
            Affine::IDENTITY,
        );
    }
}

/// A cubic of ordinary size (control box under `SPLIT_EXTENT`) too curved for 16
/// quadratics within the tolerance, seen where 16 of them are farthest from it (7 px):
/// it is flattened in blocks within the tolerance.
#[test]
fn curves_beyond_sixteen_quadratics() {
    let c = CubicBez::new(
        (-30_000.0, 50.0),
        (60_000.0, 60_000.0),
        (-60_000.0, -60_000.0),
        (30_000.0, 50.0),
    );
    let shift = Point::new(50.0, 50.0) - c.eval(0.263);
    let s = cubic([c.p0 + shift, c.p1 + shift, c.p2 + shift, c.p3 + shift].map(|p| (p.x, p.y)));
    matches_coverage("s curve", &s, Fill::NonZero, Affine::IDENTITY);
}

/// The `dash-mixed.pdf` and `dash-large-gaps.pdf` probes: a line 50 units long at
/// scale 2 (1 px wide, butt caps), dashed 10 on, 10 off, then entries far finer than a
/// pixel. The fine entries are drawn by their ink, the wide gap stays empty, and each
/// pixel's ink is within a quarter pixel of the pattern's.
#[test]
fn dense_dashes_keep_wide_gaps() {
    let mut mixed = vec![10.0, 10.0];
    mixed.extend(std::iter::repeat_n(5e-6, 10_000));
    for (name, dashes) in [
        ("mixed", mixed),
        ("large gaps", vec![10.0, 10.0, 0.025, 0.025]),
    ] {
        let mut line = BezPath::new();
        line.move_to((0.0, 25.0));
        line.line_to((50.0, 25.0));
        let stroke = Stroke::new(0.5)
            .with_caps(Cap::Butt)
            .with_dashes(0.0, dashes.iter().copied());
        let alphas = draw(|ctx| {
            ctx.set_transform(Affine::scale(2.0));
            ctx.set_stroke(stroke);
            ctx.stroke_path(&line);
        });
        // The pattern's ink over each pixel (half a unit), from its on intervals.
        let mut ink = [0.0; 100];
        let (mut at, mut i) = (0.0, 0);
        while at < 50.0 {
            let end = (at + dashes[i % dashes.len()]).min(50.0);
            if i % 2 == 0 {
                for (x, v) in ink.iter_mut().enumerate() {
                    let (a, b) = (x as f64 / 2.0, (x + 1) as f64 / 2.0);
                    *v += (end.min(b) - at.max(a)).max(0.0) * 2.0;
                }
            }
            at = end;
            i += 1;
        }
        for x in 0..100 {
            for y in [49, 50] {
                let a = f64::from(alphas[y * 100 + x]);
                let want = 255.0 * 0.5 * ink[x];
                assert!(
                    (a - want).abs() <= 0.25 * 127.5 + 2.0,
                    "{name}: ({x}, {y}) {a} vs {want}"
                );
            }
        }
        let gap = (21..40).all(|x| alphas[49 * 100 + x] == 0 && alphas[50 * 100 + x] == 0);
        assert!(gap, "{name}: the gap at x 20..40 has ink");
    }
}
