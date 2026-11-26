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
