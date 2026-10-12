// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The largest tile-aligned source endpoint remains usable by tiny filter layers.

use vello_cpu::color::palette::css::RED;
use vello_cpu::filter_effects::{Filter, FilterPrimitive};
use vello_cpu::kurbo::{Rect, Shape};
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
};

#[test]
fn maximum_source_endpoint_retains_pixels_in_both_axes() {
    for (wide_x, wide_y) in [(true, false), (false, true), (true, true)] {
        for path in [false, true] {
            for fractional in [false, true] {
                let width = if wide_x { 252 } else { 4 };
                let height = if wide_y { 252 } else { 4 };
                let dx = if wide_x { -4294967040.0_f32 } else { 0.0 };
                let dy = if wide_y { -4294967040.0_f32 } else { 0.0 };
                let x0 = if wide_x { 4294967288.0 } else { 0.0 };
                let y0 = if wide_y { 4294967288.0 } else { 0.0 };
                let inset = if fractional { 0.5 } else { 0.0 };
                let rect = Rect::new(x0 + inset, y0 + inset, x0 + 4.0 - inset, y0 + 4.0 - inset);
                let mut context = RenderContext::new_with(
                    width,
                    height,
                    RenderSettings {
                        level: Level::baseline(),
                        num_threads: 0,
                    },
                );
                context
                    .push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset { dx, dy }));
                context.set_paint(RED);
                if path {
                    context.fill_path(&rect.to_path(0.1));
                } else {
                    context.fill_rect(&rect);
                }
                context.pop_layer();
                context.flush();
                let left = if wide_x { 248 } else { 0 };
                let top = if wide_y { 248 } else { 0 };
                let covered =
                    |position: usize, start: usize| (start..start + 4).contains(&position);
                render_all_modes(&context, |x, y| {
                    let alpha = match (covered(x, left) && covered(y, top), fractional) {
                        (false, _) => 0,
                        (true, false) => 255,
                        // A half-pixel inset covers a quarter of each corner pixel and
                        // half of each edge pixel of the 4x4 square.
                        (true, true) => {
                            let edges = [x - left, y - top]
                                .into_iter()
                                .filter(|&offset| offset == 0 || offset == 3)
                                .count();
                            [255, 128, 64][edges]
                        }
                    };
                    [alpha, 0, 0, alpha]
                });
            }
        }
    }
}

fn render_all_modes(context: &RenderContext, expected: impl Fn(usize, usize) -> [u8; 4]) {
    for mode in [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ] {
        let mut pixels = Pixmap::new(context.width(), context.height());
        context.render_with(
            &mut pixels,
            &mut Resources::new(),
            RasterizerSettings {
                render_mode: mode,
                ..Default::default()
            },
        );
        let width = usize::from(context.width());
        for (index, pixel) in pixels.data_as_u8_slice().chunks_exact(4).enumerate() {
            let (x, y) = (index % width, index / width);
            assert_eq!(pixel, expected(x, y), "{mode:?}: pixel ({x},{y})");
        }
    }
}

#[test]
fn rejected_nested_filter_preserves_context_and_reset_restores_rendering() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use vello_cpu::filter_effects::EdgeMode;
    let mut context = RenderContext::new_with(
        236,
        16,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: -4294967040.0,
        dy: 0.0,
    }));
    context.set_paint(RED);
    context.fill_rect(&Rect::new(4294967272.0, 4.0, 4294967284.0, 12.0));
    let before = format!("{context:?}");
    for _ in 0..2 {
        // The blur halo around the visible source reaches past u32 source coordinates.
        let error = catch_unwind(AssertUnwindSafe(|| {
            context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: 1.5,
                edge_mode: EdgeMode::None,
            }));
        }))
        .expect_err("the nested halo exceeds u32 source coordinates");
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied());
        assert_eq!(
            message,
            Some("filter source viewport exceeds u32 coordinate domain")
        );
        assert_eq!(
            format!("{context:?}"),
            before,
            "a rejected push must not change any layer, viewport, or transform stack"
        );
    }
    context.pop_layer();
    context.flush();
    render_all_modes(&context, |x, y| {
        if x >= 232 && (4..12).contains(&y) {
            [255, 0, 0, 255]
        } else {
            [0; 4]
        }
    });

    context.reset();
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: 0.0,
        dy: 0.0,
    }));
    context.set_paint(RED);
    context.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
    context.pop_layer();
    context.flush();
    render_all_modes(&context, |x, y| {
        if x < 4 && y < 4 {
            [255, 0, 0, 255]
        } else {
            [0; 4]
        }
    });
}
