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
    let modes = [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ];
    for mode in modes {
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
                    let rect =
                        Rect::new(x0 + inset, y0 + inset, x0 + 4.0 - inset, y0 + 4.0 - inset);
                    let mut context = RenderContext::new_with(
                        width,
                        height,
                        RenderSettings {
                            level: Level::baseline(),
                            num_threads: 0,
                        },
                    );
                    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
                        dx,
                        dy,
                    }));
                    context.set_paint(RED);
                    if path {
                        context.fill_path(&rect.to_path(0.1));
                    } else {
                        context.fill_rect(&rect);
                    }
                    context.pop_layer();
                    context.flush();
                    let mut pixmap = Pixmap::new(width, height);
                    context.render_with(
                        &mut pixmap,
                        &mut Resources::new(),
                        RasterizerSettings {
                            render_mode: mode,
                            ..Default::default()
                        },
                    );
                    let left = if wide_x { 248.0_f64 } else { 0.0 };
                    let top = if wide_y { 248.0_f64 } else { 0.0 };
                    for (index, pixel) in pixmap.data_as_u8_slice().chunks_exact(4).enumerate() {
                        let x = (index % usize::from(width)) as f64;
                        let y = (index / usize::from(width)) as f64;
                        let coverage_x = ((x + 1.0).min(left + 4.0 - inset) - x.max(left + inset))
                            .clamp(0.0, 1.0);
                        let coverage_y =
                            ((y + 1.0).min(top + 4.0 - inset) - y.max(top + inset)).clamp(0.0, 1.0);
                        let alpha = (coverage_x * coverage_y * 255.0 + 0.5) as u8;
                        assert_eq!(
                            pixel,
                            [alpha, 0, 0, alpha],
                            "mode={mode:?} x={wide_x} y={wide_y} path={path} fractional={fractional} pixel=({x},{y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn rejected_filter_halo_pop_preserves_context_and_reset_restores_rendering() {
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
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 1.5,
        edge_mode: EdgeMode::None,
    }));
    context.set_paint(RED);
    context.fill_rect(&Rect::new(4294967272.0, 4.0, 4294967284.0, 12.0));
    for record_after_failure in [false, true] {
        if record_after_failure {
            context.fill_rect(&Rect::new(4294967272.0, 4.0, 4294967276.0, 8.0));
        }
        let before = format!("{context:?}");
        for _ in 0..2 {
            let error = catch_unwind(AssertUnwindSafe(|| context.pop_layer()))
                .expect_err("the full halo exceeds u32 source coordinates");
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied());
            assert_eq!(message, Some("source rectangle overflow"));
            assert_eq!(
                format!("{context:?}"),
                before,
                "a failed pop must not advance any layer, viewport, or transform stack"
            );
        }
    }
    context.reset();
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: 0.0,
        dy: 0.0,
    }));
    context.set_paint(RED);
    context.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
    context.pop_layer();
    context.flush();
    for mode in [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ] {
        let mut pixels = Pixmap::new(236, 16);
        context.render_with(
            &mut pixels,
            &mut Resources::new(),
            RasterizerSettings {
                render_mode: mode,
                ..Default::default()
            },
        );
        for (index, pixel) in pixels.data_as_u8_slice().chunks_exact(4).enumerate() {
            let expected = if index % 236 < 4 && index / 236 < 4 {
                [255, 0, 0, 255]
            } else {
                [0; 4]
            };
            assert_eq!(pixel, expected, "{mode:?}: pixel {index}");
        }
    }
}
