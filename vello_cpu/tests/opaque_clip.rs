// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel parity across clip snapshots, worker batches and context reuse.

#![cfg(feature = "multithreading")]

use vello_cpu::color::palette::css;
use vello_cpu::kurbo::{BezPath, Rect, Shape, Stroke};
use vello_cpu::peniko::Fill;
use vello_cpu::{Level, Pixmap, RenderContext, RenderSettings, Resources};

#[test]
fn opaque_clip_metadata_survives_worker_batches_pop_and_reset() {
    let mut reference = None;
    for num_threads in [0, 1, 4] {
        let mut context = RenderContext::new_with(
            100,
            100,
            RenderSettings {
                level: Level::baseline(),
                num_threads,
            },
        );
        let mut resources = Resources::new();
        let mut target = Pixmap::new(100, 100);
        for _ in 0..2 {
            context.set_paint(css::BLACK);
            context.set_stroke(Stroke::new(0.75));
            context.push_clip_path(&Rect::new(3.25, 4.5, 94.75, 95.5).to_path(0.1));
            // Both opaque interior and fractional boundary strokes cross worker batches.
            for i in 0..800 {
                let x = 3.25 + f64::from(i % 80);
                let y = 4.5 + f64::from(i / 80) * 8.0;
                let mut path = BezPath::new();
                path.move_to((x, y));
                path.line_to((x + 10.5, y + 6.25));
                context.stroke_path(&path);
            }
            context.set_fill_rule(Fill::EvenOdd);
            let mut hole = Rect::new(9.5, 8.25, 88.5, 87.75).to_path(0.1);
            hole.extend(Rect::new(35.25, 30.5, 65.75, 70.25).to_path(0.1).iter());
            context.push_clip_path(&hole);
            context.set_paint(css::RED);
            context.fill_rect(&Rect::new(20.25, 15.5, 80.75, 85.25));
            context.pop_clip();
            context.set_paint(css::BLUE);
            context.fill_rect(&Rect::new(70.25, 72.5, 100.75, 99.25));
            context.pop_clip();
            // An overlay drawn after page clipping must remain visible outside it.
            context.set_paint(css::GREEN);
            context.fill_rect(&Rect::new(0.25, 0.5, 5.75, 5.25));
            context.flush();
            context.render(&mut target, &mut resources);
            if let Some(ref expected) = reference {
                assert_eq!(target.data(), expected, "threads={num_threads}");
            } else {
                reference = Some(target.data().to_vec());
            }
            assert_eq!(target.data()[101], css::GREEN.premultiply().to_rgba8());
            context.reset();
        }
    }
}
