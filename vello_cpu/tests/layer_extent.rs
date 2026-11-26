// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A layer without a clip leaves every pixel it draws nothing into as it was, whatever
//! its blend mode and opacity, unless that blend mode is destructive.

use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::kurbo::Rect;
use vello_cpu::peniko::{BlendMode, Compose, Mix};
use vello_cpu::{Pixmap, RenderContext, Resources};

const WIDTH: u16 = 64;
const HEIGHT: u16 = 8;

/// A backdrop of varied colours and opacities, one per pixel.
fn backdrop(ctx: &mut RenderContext) {
    let mut seed = 0x2545_f491_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as u8
    };
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(
                next(),
                next(),
                next(),
                next(),
            ));
            let (x, y) = (f64::from(x), f64::from(y));
            ctx.fill_rect(&Rect::new(x, y, x + 1.0, y + 1.0));
        }
    }
}

fn render(layer: Option<(BlendMode, f32)>) -> Pixmap {
    let mut ctx = RenderContext::new(WIDTH, HEIGHT);
    backdrop(&mut ctx);
    if let Some((blend_mode, opacity)) = layer {
        ctx.push_layer(None, Some(blend_mode), Some(opacity), None, None);
        ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(200, 40, 90, 180));
        ctx.fill_rect(&Rect::new(1.5, 1.25, 6.5, 6.75));
        ctx.pop_layer();
    }
    let mut pixmap = Pixmap::new(WIDTH, HEIGHT);
    ctx.flush();
    ctx.render(&mut pixmap, &mut Resources::default());
    pixmap
}

#[test]
fn a_layer_leaves_the_pixels_it_does_not_draw_in_unchanged() {
    let without = render(None);
    let mixes = [
        Mix::Normal,
        Mix::Multiply,
        Mix::Screen,
        Mix::Overlay,
        Mix::Darken,
        Mix::Lighten,
        Mix::ColorDodge,
        Mix::ColorBurn,
        Mix::HardLight,
        Mix::SoftLight,
        Mix::Difference,
        Mix::Exclusion,
        Mix::Hue,
        Mix::Saturation,
        Mix::Color,
        Mix::Luminosity,
    ];
    let composes = [
        Compose::SrcOver,
        Compose::DestOver,
        Compose::Dest,
        Compose::SrcAtop,
        Compose::DestOut,
        Compose::Xor,
        Compose::Plus,
        Compose::PlusLighter,
    ];
    for mix in mixes {
        for compose in composes {
            for opacity in [1.0, 0.5] {
                let blend_mode = BlendMode::new(mix, compose);
                let with = render(Some((blend_mode, opacity)));
                for y in 0..HEIGHT {
                    // The layer draws within the first two tiles.
                    for x in 8..WIDTH {
                        assert_eq!(
                            with.sample(x, y),
                            without.sample(x, y),
                            "{mix:?} {compose:?} {opacity} at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn isolated_layer_clip_is_independent_of_fractional_draw_clip() {
    use vello_cpu::kurbo::Shape;
    use vello_cpu::{
        Level, RenderSettings,
        color::palette::css::{BLUE, GREEN, RED},
    };
    let full = Rect::new(0., 0., 8., 8.);
    let full_path = full.to_path(0.1);
    for num_threads in [
        0,
        #[cfg(feature = "multithreading")]
        2,
    ] {
        for count in 0..=2 {
            let draw = |explicit: bool| {
                let mut ctx = RenderContext::new_with(
                    8,
                    8,
                    RenderSettings {
                        level: Level::baseline(),
                        num_threads,
                    },
                );
                ctx.set_paint(BLUE);
                ctx.fill_rect(&full);
                ctx.push_clip_path(&Rect::new(2.5, 0., 6., 8.).to_path(0.1));
                let blend = (count == 0).then(|| BlendMode::new(Mix::Normal, Compose::Clear));
                ctx.push_layer(explicit.then_some(&full_path), blend, None, None, None);
                ctx.set_paint(RED);
                for _ in 0..count {
                    ctx.fill_rect(&full);
                }
                ctx.pop_layer();
                ctx.fill_rect(&Rect::new(0., 5., 8., 6.));
                ctx.pop_clip();
                ctx.set_paint(GREEN);
                ctx.fill_rect(&Rect::new(0., 6., 8., 8.));
                ctx.flush();
                let mut image = Pixmap::new(8, 8);
                ctx.render(&mut image, &mut Resources::new());
                image
            };
            assert_eq!(
                draw(true).data(),
                draw(false).data(),
                "threads {num_threads}, draws {count}"
            );
        }
    }
}
