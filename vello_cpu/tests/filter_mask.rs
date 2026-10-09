// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Physical scene masks retain their coordinates through tight and nested filter targets.

use std::sync::Arc;
use vello_cpu::color::palette::css::RED;
use vello_cpu::filter_effects::{Filter, FilterPrimitive};
use vello_cpu::kurbo::Rect;
use vello_cpu::peniko::ImageSampler;
use vello_cpu::{
    Image, ImageSource, Level, Mask, Pixmap, RasterizerSettings, RenderContext, RenderMode,
    RenderSettings, Resources,
};

fn modes() -> impl Iterator<Item = RenderMode> {
    [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ]
    .into_iter()
}

fn offset(distance: f32, vertical: bool) -> Filter {
    let (dx, dy) = if vertical {
        (0.0, distance)
    } else {
        (distance, 0.0)
    };
    Filter::from_primitive(FilterPrimitive::Offset { dx, dy })
}

fn render(
    vertical: bool,
    layer_mask: bool,
    nested: bool,
    outside: Option<f64>,
    indexed: bool,
    mode: RenderMode,
) -> Pixmap {
    let (width, height) = if vertical { (4, 16) } else { (16, 4) };
    let mut ctx = RenderContext::new_with(
        width,
        height,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    let data = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let major = if vertical { y } else { x };
                if outside.is_some() || (8..12).contains(&major) {
                    255
                } else {
                    0
                }
            })
        })
        .collect();
    let mask = Mask::from_parts(data, width, height);
    if nested {
        ctx.push_filter_layer(offset(-70_000.0, vertical));
        ctx.push_filter_layer(offset(70_000.0, vertical));
    } else {
        ctx.push_filter_layer(offset(
            outside.map_or(0.0, |start| (4.0 - start) as f32),
            vertical,
        ));
    }
    if layer_mask {
        ctx.push_mask_layer(mask);
    } else {
        ctx.set_mask(mask);
    }
    if indexed {
        let mut source = Pixmap::new(1, 1);
        source.data_mut().fill(RED.premultiply().to_rgba8());
        ctx.set_paint(Image {
            image: ImageSource::Pixmap(Arc::new(source)),
            sampler: ImageSampler::default(),
        });
    } else {
        ctx.set_paint(RED);
    }
    let start = outside.unwrap_or(8.0);
    let end = start + 4.0;
    let rect = if vertical {
        Rect::new(0.0, start, 4.0, end)
    } else {
        Rect::new(start, 0.0, end, 4.0)
    };
    ctx.fill_rect(&rect);
    if layer_mask {
        ctx.pop_layer();
    } else {
        ctx.reset_mask();
    }
    ctx.pop_layer();
    if nested {
        ctx.pop_layer();
    }
    finish(&mut ctx, mode)
}

fn finish(ctx: &mut RenderContext, mode: RenderMode) -> Pixmap {
    ctx.flush();
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_with(
        &mut pixmap,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: mode,
            ..Default::default()
        },
    );
    pixmap
}

fn verify(layer_mask: bool, nested: bool, outside: Option<f64>) {
    for mode in modes() {
        for (vertical, indexed) in [(false, false), (true, false), (false, true), (true, true)] {
            let pixmap = render(vertical, layer_mask, nested, outside, indexed, mode);
            for y in 0..pixmap.height() {
                for x in 0..pixmap.width() {
                    let major = if vertical { y } else { x };
                    let expected = if outside.is_none() && (8..12).contains(&major) {
                        [255, 0, 0, 255]
                    } else {
                        [0; 4]
                    };
                    assert_eq!(
                        pixmap.sample(x, y).to_u8_array(),
                        expected,
                        "layer={layer_mask} nested={nested} outside={outside:?} indexed={indexed} mode={mode:?} vertical={vertical} pixel=({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn draw_mask_uses_scene_coordinates_in_tight_filter_buffer() {
    verify(false, false, None);
}
#[test]
fn layer_mask_uses_scene_coordinates_in_tight_filter_buffer() {
    verify(true, false, None);
}
#[test]
fn nested_filter_masks_compensate_cumulative_source_shifts() {
    verify(false, true, None);
    verify(true, true, None);
}
#[test]
fn filter_source_outside_physical_mask_is_transparent() {
    for outside in [Some(20.0), Some(-12.0)] {
        verify(false, false, outside);
        verify(true, false, outside);
    }
}

#[test]
fn filter_layer_mask_uses_parent_coordinates_and_reset_clears_offsets() {
    for mode in modes() {
        for vertical in [false, true] {
            for outer in [-4.0, 4.0] {
                let (width, height) = if vertical { (4, 16) } else { (16, 4) };
                let mut ctx = RenderContext::new_with(
                    width,
                    height,
                    RenderSettings {
                        level: Level::baseline(),
                        num_threads: 0,
                    },
                );
                let band_start = (8.0 - outer) as u16;
                let data = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let major = if vertical { y } else { x };
                            if (band_start..band_start + 4).contains(&major) {
                                128
                            } else {
                                0
                            }
                        })
                    })
                    .collect();
                ctx.push_filter_layer(offset(outer, vertical));
                ctx.push_layer(
                    None,
                    None,
                    None,
                    Some(Mask::from_parts(data, width, height)),
                    Some(offset(-outer, vertical)),
                );
                ctx.set_paint(RED);
                ctx.fill_rect(&if vertical {
                    Rect::new(0.0, 8.0, 4.0, 12.0)
                } else {
                    Rect::new(8.0, 0.0, 12.0, 4.0)
                });
                ctx.pop_layer();
                ctx.pop_layer();
                let actual = finish(&mut ctx, mode);
                let repeated = finish(&mut ctx, mode);
                assert_eq!(
                    actual.data(),
                    repeated.data(),
                    "replaying recorded masks must retain offsets"
                );
                for y in 0..height {
                    for x in 0..width {
                        let major = if vertical { y } else { x };
                        let expected = if (8..12).contains(&major) {
                            [128, 0, 0, 128]
                        } else {
                            [0; 4]
                        };
                        assert_eq!(
                            actual.sample(x, y).to_u8_array(),
                            expected,
                            "filter props mask, outer={outer}, mode={mode:?}, vertical={vertical}, x={x}, y={y}"
                        );
                    }
                }
                ctx.reset();
                ctx.set_paint(RED);
                let data = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let major = if vertical { y } else { x };
                            if (2..6).contains(&major) { 128 } else { 0 }
                        })
                    })
                    .collect();
                ctx.set_mask(Mask::from_parts(data, width, height));
                ctx.fill_rect(&Rect::new(0.0, 0.0, f64::from(width), f64::from(height)));
                let actual = finish(&mut ctx, mode);
                for y in 0..height {
                    for x in 0..width {
                        let major = if vertical { y } else { x };
                        let expected = if (2..6).contains(&major) {
                            [128, 0, 0, 128]
                        } else {
                            [0; 4]
                        };
                        assert_eq!(
                            actual.sample(x, y).to_u8_array(),
                            expected,
                            "reset mask, outer={outer}, mode={mode:?}, vertical={vertical}, x={x}, y={y}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn blur_of_masked_source_matches_blur_of_the_visible_geometry() {
    use vello_cpu::filter_effects::EdgeMode;

    for mode in modes() {
        for layer_mask in [false, true] {
            let render_case = |masked: bool| {
                let mut ctx = RenderContext::new_with(
                    32,
                    16,
                    RenderSettings {
                        level: Level::baseline(),
                        num_threads: 0,
                    },
                );
                ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                    std_deviation: 2.0,
                    edge_mode: EdgeMode::None,
                }));
                if masked {
                    let data = (0..16)
                        .flat_map(|_| (0..32).map(|x| if (12..20).contains(&x) { 255 } else { 0 }))
                        .collect();
                    let mask = Mask::from_parts(data, 32, 16);
                    if layer_mask {
                        ctx.push_mask_layer(mask);
                    } else {
                        ctx.set_mask(mask);
                    }
                }
                ctx.set_paint(RED);
                let bounds = if masked { (8.0, 24.0) } else { (12.0, 20.0) };
                ctx.fill_rect(&Rect::new(bounds.0, 4.0, bounds.1, 12.0));
                if masked {
                    if layer_mask {
                        ctx.pop_layer();
                    } else {
                        ctx.reset_mask();
                    }
                }
                ctx.pop_layer();
                finish(&mut ctx, mode)
            };
            let actual = render_case(true);
            let expected = render_case(false);
            assert!(
                expected.data().iter().any(|pixel| pixel.a != 0),
                "oracle must contain visible blur"
            );
            assert_eq!(
                actual.data(),
                expected.data(),
                "masked blur, layer={layer_mask}, mode={mode:?}"
            );
        }
    }
}
