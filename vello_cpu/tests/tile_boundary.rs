// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel coverage around tile boundary crossings.

use vello_cpu::color::palette::css::{BLACK, WHITE};
use vello_cpu::kurbo::{Affine, BezPath, Rect};
use vello_cpu::{Level, Pixmap, RenderContext, RenderSettings, Resources};

const WORKER_COUNTS: &[u16] = &[
    0,
    #[cfg(feature = "multithreading")]
    2,
];

fn check_edge(draw: impl Fn(&mut RenderContext)) {
    for &num_threads in WORKER_COUNTS {
        let mut ctx = RenderContext::new_with(
            312,
            628,
            RenderSettings {
                level: Level::baseline(),
                num_threads,
            },
        );
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 312.0, 628.0));
        ctx.set_paint(BLACK);
        draw(&mut ctx);
        ctx.flush();
        let mut pixmap = Pixmap::new(312, 628);
        ctx.render(&mut pixmap, &mut Resources::new());
        for y in 412..624 {
            let row = &pixmap.data()[y * 312..][146..154];
            assert_eq!(
                core::array::from_fn(|x| row[x].r),
                [255, 255, 0, 0, 0, 0, 0, 0],
                "y={y}, threads={num_threads}"
            );
        }
    }
}

#[test]
fn near_vertical_quad() {
    for dx in [-0.00002, 0.00002] {
        check_edge(|ctx| {
            let mut path = BezPath::new();
            path.move_to((148.0 - dx, 411.0));
            path.line_to((148.0 + dx, 625.0));
            path.line_to((310.0, 625.0));
            path.line_to((310.0, 411.0));
            path.close_path();
            ctx.fill_path(&path);
        });
    }
}

#[test]
fn near_right_angle_rect() {
    check_edge(|ctx| {
        ctx.set_transform(Affine::new([
            -1.53e-5,
            212.6376953,
            -212.6376953,
            -1.53e-5,
            228.7763672,
            411.4482422,
        ]));
        ctx.fill_rect(&Rect::new(
            -0.0021080366,
            -0.3819813446,
            1.0042986634,
            0.3798779286,
        ));
    });
}

#[test]
fn edge_near_tile_corner_keeps_fractional_coverage() {
    let mut ctx = RenderContext::new(140, 28);
    let mut path = BezPath::new();
    path.move_to((32.89 * 4.0, 0.9 * 4.0));
    path.line_to((33.5 * 4.0, 7.0 * 4.0));
    path.line_to((140.0, 28.0));
    path.line_to((140.0, 3.6));
    path.close_path();
    ctx.set_paint(BLACK);
    ctx.fill_path(&path);
    ctx.flush();
    let mut pixmap = Pixmap::new(140, 28);
    ctx.render(&mut pixmap, &mut Resources::new());
    for (y, alpha) in (8..12).zip([242, 217, 191, 166]) {
        let row = &pixmap.data()[y * 140..];
        // The edge passes x = 132 at the top of the tile and advances
        // 0.1 pixel per row. Only column 132 has partial coverage.
        assert!(row[128..132].iter().all(|p| p.a == 0));
        assert_eq!(row[132].a, alpha);
        assert!(row[133..140].iter().all(|p| p.a == 255));
    }
}

fn render_quad(points: [(f64, f64); 4], reverse: bool) -> Pixmap {
    let mut path = BezPath::new();
    path.move_to(points[0]);
    for i in 1..4 {
        path.line_to(points[if reverse { 4 - i } else { i }]);
    }
    path.close_path();
    let mut ctx = RenderContext::new(200, 120);
    ctx.set_paint(BLACK);
    ctx.fill_path(&path);
    ctx.flush();
    let mut pixmap = Pixmap::new(200, 120);
    ctx.render(&mut pixmap, &mut Resources::new());
    pixmap
}

#[test]
fn distant_endpoint_keeps_diagonal_coverage() {
    let far = 2_f64.powi(60);
    for reverse in [false, true] {
        let pixmap = render_quad(
            [(20., 100.), (far, -far), (200., -far), (200., 100.)],
            reverse,
        );
        for (y, row) in pixmap.data().as_chunks::<200>().0.iter().enumerate() {
            for (x, pixel) in row.iter().enumerate() {
                let alpha = if y >= 100 || x + y < 119 {
                    0
                } else if x + y == 119 {
                    128
                } else {
                    255
                };
                assert_eq!(pixel.a, alpha, "({x}, {y}), reverse={reverse}");
            }
        }
    }
}

#[test]
fn exact_corner_keeps_diagonal_coverage() {
    for reverse in [false, true] {
        let pixmap = render_quad([(0., 1.), (72., 67.), (100., 67.), (100., 1.)], reverse);
        // At (60, 56) the edge advances 12/11 pixels per row. The pixel
        // above-left is 13/24 covered and the pixel below-right is 11/24 covered.
        for (y, expected) in [(55, [138, 255]), (56, [0, 117])] {
            let row = &pixmap.data()[y * 200..];
            assert_eq!(
                [row[59].a, row[60].a],
                expected,
                "row {y}, reverse={reverse}"
            );
        }
    }
}

#[test]
fn subnormal_top_edge() {
    let s = f64::from(f32::from_bits(1));
    let mut path = BezPath::new();
    path.move_to((1., s));
    path.line_to((5., 2. * s));
    path.line_to((5., 4.));
    path.line_to((1., 4.));
    path.close_path();
    for &num_threads in WORKER_COUNTS {
        let mut ctx = RenderContext::new_with(
            8,
            4,
            RenderSettings {
                level: Level::baseline(),
                num_threads,
            },
        );
        ctx.set_paint(BLACK);
        ctx.fill_path(&path);
        ctx.flush();
        let mut pixmap = Pixmap::new(8, 4);
        ctx.render(&mut pixmap, &mut Resources::new());
        assert_eq!(
            [pixmap.data()[0].a, pixmap.data()[1].a],
            [0, 255],
            "threads={num_threads}"
        );
    }
}
