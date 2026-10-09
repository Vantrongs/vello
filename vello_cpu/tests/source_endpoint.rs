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
