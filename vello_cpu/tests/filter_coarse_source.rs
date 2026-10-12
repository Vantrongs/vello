// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A blur wide enough to read its source at a coarser resolution still sees
//! every pixel of that source.

use std::sync::Arc;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Rect};
use vello_cpu::peniko::{ImageQuality, ImageSampler};
use vello_cpu::{
    Image, ImageSource, Level, Pixmap, RasterizerSettings, RenderContext, RenderMode,
    RenderSettings, Resources,
};

const VIEW: u16 = 64;
/// How far past the view the texture reaches, beyond the blur's reach.
const MARGIN: u16 = 450;
const SIZE: u16 = VIEW + 2 * MARGIN;

fn texture(value: impl Fn(u16, u16) -> [u8; 4]) -> Pixmap {
    let mut pixmap = Pixmap::new(SIZE, SIZE);
    for (index, pixel) in pixmap.data_mut().iter_mut().enumerate() {
        let coordinate = |value: usize| u16::try_from(value).unwrap();
        let (x, y) = (
            coordinate(index % usize::from(SIZE)),
            coordinate(index / usize::from(SIZE)),
        );
        [pixel.r, pixel.g, pixel.b, pixel.a] = value(x, y);
    }
    pixmap
}

/// An opaque texture whose pixels alternate between black and white.
fn stripes(white: impl Fn(u16, u16) -> bool) -> Pixmap {
    texture(|x, y| {
        let value = if white(x, y) { 255 } else { 0 };
        [value, value, value, 255]
    })
}

fn blur(std_deviation: f32) -> Filter {
    Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    })
}

fn render(
    width: u16,
    height: u16,
    mode: RenderMode,
    draw: impl FnOnce(&mut RenderContext),
) -> Pixmap {
    let mut ctx = RenderContext::new_with(
        width,
        height,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    draw(&mut ctx);
    ctx.flush();
    let mut image = Pixmap::new(width, height);
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

fn paint(texture: &Pixmap, sampler: ImageSampler) -> Image {
    Image {
        image: ImageSource::Pixmap(Arc::new(texture.clone())),
        sampler,
    }
}

/// Something to draw, given the transform to draw it with.
type Draw = Box<dyn Fn(&mut RenderContext, Affine)>;

/// Fill the area from `-MARGIN` to `VIEW + MARGIN` with `texture`, sampled
/// with `sampler` and transformed by `transform` about the center of the view.
fn draw_texture(texture: &Pixmap, sampler: ImageSampler, transform: Affine) -> Draw {
    let paint = paint(texture, sampler);
    Box::new(move |ctx, base| {
        let margin = f64::from(MARGIN);
        let center = f64::from(VIEW) / 2.0 + margin;
        ctx.set_transform(base * Affine::translate((-margin, -margin)));
        ctx.set_paint_transform(
            Affine::translate((center, center)) * transform * Affine::translate((-center, -center)),
        );
        ctx.set_paint(paint.clone());
        let size = f64::from(SIZE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, size, size));
    })
}

fn sampler(quality: ImageQuality) -> ImageSampler {
    ImageSampler {
        quality,
        ..ImageSampler::default()
    }
}

#[test]
fn wide_blur_of_a_fine_texture_is_its_average() {
    let textures = [
        ("columns", stripes(|x, _| x % 2 == 0)),
        ("rows", stripes(|_, y| y % 2 == 1)),
        ("checkers", stripes(|x, y| (x + y) % 2 == 0)),
    ];
    for (name, texture) in &textures {
        for quality in [ImageQuality::Low, ImageQuality::Medium, ImageQuality::High] {
            for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
                let draw = draw_texture(texture, sampler(quality), Affine::IDENTITY);
                let image = render(VIEW, VIEW, mode, |ctx| {
                    ctx.push_filter_layer(blur(100.0));
                    draw(ctx, Affine::IDENTITY);
                    ctx.pop_layer();
                });
                for pixel in image.data() {
                    assert!(
                        pixel.a == 255
                            && [pixel.r, pixel.g, pixel.b]
                                .iter()
                                .all(|value| value.abs_diff(128) <= 3),
                        "{name}, {quality:?}, {mode:?}: {pixel:?}"
                    );
                }
            }
        }
    }
}
