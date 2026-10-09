// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter padding must cover the device-space kernel before source geometry is culled.

use vello_cpu::color::palette::css::{BLUE, RED};
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Rect};
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
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

#[derive(Clone, Copy, Debug)]
enum Effect {
    Blur,
    Shadow,
    ShadowOnly,
}

fn filter(effect: Effect, sigma: f32) -> Filter {
    Filter::from_primitive(match effect {
        Effect::Blur => FilterPrimitive::GaussianBlur {
            std_deviation: sigma,
            edge_mode: EdgeMode::None,
        },
        Effect::Shadow => FilterPrimitive::DropShadow {
            dx: 2.0,
            dy: -2.0,
            std_deviation: sigma,
            color: BLUE,
            edge_mode: EdgeMode::None,
        },
        Effect::ShadowOnly => FilterPrimitive::DropShadowOnly {
            dx: 2.0,
            dy: -2.0,
            std_deviation: sigma,
            color: BLUE,
            edge_mode: EdgeMode::None,
        },
    })
}

fn render(
    size: u16,
    shift: f64,
    transform: Affine,
    filter: Filter,
    vertical: bool,
    mode: RenderMode,
    outside: bool,
) -> Pixmap {
    let mut ctx = RenderContext::new_with(
        size,
        size,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    ctx.set_transform(transform);
    ctx.push_filter_layer(filter);
    // The filter retains its own transform. Geometry is subsequently specified
    // directly in device pixels, so singular filter transforms cannot erase it.
    ctx.set_transform(Affine::translate((shift, shift)));
    ctx.set_paint(RED);
    if outside {
        for start in [-8.0, 12.0] {
            let rect = if vertical {
                Rect::new(4.0, start, 8.0, start + 4.0)
            } else {
                Rect::new(start, 4.0, start + 4.0, 8.0)
            };
            ctx.fill_rect(&rect);
        }
    } else {
        ctx.fill_rect(&Rect::new(0.25, 0.25, 3.75, 3.75));
    }
    ctx.pop_layer();
    ctx.flush();
    let mut image = Pixmap::new(size, size);
    ctx.render_with(
        &mut image,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: mode,
            ..Default::default()
        },
    );
    image
}

#[test]
fn transformed_isotropic_halo_retains_source_outside_the_frame() {
    for mode in modes() {
        for vertical in [false, true] {
            let transforms = if vertical {
                [
                    Affine::scale_non_uniform(2.0, 0.0),
                    Affine::scale_non_uniform(1.5, 0.5),
                    Affine::new([1.0, 1.5, 0.0, 1.0, 0.0, 0.0]),
                ]
            } else {
                [
                    Affine::scale_non_uniform(0.0, 2.0),
                    Affine::scale_non_uniform(0.5, 1.5),
                    Affine::new([1.0, 0.0, 1.5, 1.0, 0.0, 0.0]),
                ]
            };
            for transform in transforms {
                for effect in [Effect::Blur, Effect::Shadow, Effect::ShadowOnly] {
                    let actual = render(
                        12,
                        0.0,
                        transform,
                        filter(effect, 8.0),
                        vertical,
                        mode,
                        true,
                    );
                    // Both rectangles are inside the reference viewport. Translation
                    // and any omitted empty left/top padding are multiples of eight,
                    // preserving phase through at most three decimation levels.
                    let reference = render(
                        140,
                        64.0,
                        transform,
                        filter(effect, 8.0),
                        vertical,
                        mode,
                        true,
                    );
                    let mut expected_alpha = 0u64;
                    for y in 0..12usize {
                        for x in 0..12usize {
                            let expected_index = ((y + 64) * 140 + x + 64) * 4;
                            let expected =
                                &reference.data_as_u8_slice()[expected_index..expected_index + 4];
                            expected_alpha += u64::from(expected[3]);
                            let actual_index = (y * 12 + x) * 4;
                            assert_eq!(
                                &actual.data_as_u8_slice()[actual_index..actual_index + 4],
                                expected,
                                "transform={transform:?}, effect={effect:?}, vertical={vertical}, mode={mode:?}, pixel=({x},{y})"
                            );
                        }
                    }
                    assert_ne!(
                        expected_alpha, 0,
                        "empty halo reference for {transform:?}, {effect:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn tiny_filter_scale_matches_its_effective_device_sigma() {
    let sigma = 1e11_f32;
    // The public parameter is f32, so preserve its actual value before applying
    // the f64 affine scale rather than rounding an ideal decimal 0.1.
    let effective_sigma = (f64::from(sigma) * 1e-12) as f32;
    for mode in modes() {
        let actual = render(
            12,
            0.0,
            Affine::scale(1e-12),
            filter(Effect::Blur, sigma),
            false,
            mode,
            false,
        );
        let reference = render(
            12,
            0.0,
            Affine::IDENTITY,
            filter(Effect::Blur, effective_sigma),
            false,
            mode,
            false,
        );
        assert!(reference.data().iter().any(|p| p.a != 0));
        assert_eq!(
            actual.data_as_u8_slice(),
            reference.data_as_u8_slice(),
            "{mode:?}"
        );
    }
}
