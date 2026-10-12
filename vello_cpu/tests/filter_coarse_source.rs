// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A blur wide enough to read its source at a coarser resolution still sees
//! every pixel of that source.

use std::sync::Arc;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Rect};
use vello_cpu::peniko::{Extend, ImageQuality, ImageSampler};
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

/// A texture of smooth color fields with per-pixel noise, like a photo.
fn photo() -> Pixmap {
    texture(|x, y| {
        let noise = |seed: u32| {
            let mut hash = u32::from(x).wrapping_mul(0x9e37_79b9)
                ^ u32::from(y).wrapping_mul(0x85eb_ca6b)
                ^ seed;
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x2c1b_3c6d);
            hash ^= hash >> 12;
            f64::from(hash % 97) - 48.0
        };
        let (x, y) = (f64::from(x), f64::from(y));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the value is clamped to `u8`"
        )]
        let field = |a: f64, b: f64, seed: u32| {
            (128.0 + 80.0 * (x * a).sin() * (y * b).cos() + noise(seed)).clamp(0.0, 255.0) as u8
        };
        [
            field(0.013, 0.021, 1),
            field(0.029, 0.007, 2),
            field(0.011, 0.017, 3),
            255,
        ]
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

/// Return the largest difference of a color channel between `a` and `b`.
fn largest_difference(a: &Pixmap, b: &Pixmap) -> u8 {
    a.data()
        .iter()
        .zip(b.data())
        .flat_map(|(a, b)| {
            [
                a.r.abs_diff(b.r),
                a.g.abs_diff(b.g),
                a.b.abs_diff(b.b),
                a.a.abs_diff(b.a),
            ]
        })
        .max()
        .unwrap()
}

/// Return `pixmap` with each pixel averaging a square of `scale` × `scale` of its pixels.
fn decimate(pixmap: &Pixmap, scale: u16) -> Pixmap {
    let (width, height) = (pixmap.width() / scale, pixmap.height() / scale);
    let mut decimated = Pixmap::new(width, height);
    let area = u32::from(scale) * u32::from(scale);
    for y in 0..height {
        for x in 0..width {
            let mut sums = [0_u32; 4];
            for dy in 0..scale {
                for dx in 0..scale {
                    let pixel = pixmap.sample(x * scale + dx, y * scale + dy);
                    for (sum, value) in sums.iter_mut().zip([pixel.r, pixel.g, pixel.b, pixel.a]) {
                        *sum += u32::from(value);
                    }
                }
            }
            let pixel =
                &mut decimated.data_mut()[usize::from(y) * usize::from(width) + usize::from(x)];
            [pixel.r, pixel.g, pixel.b, pixel.a] =
                sums.map(|sum| u8::try_from((sum + area / 2) / area).unwrap());
        }
    }
    decimated
}

/// Return the largest difference between `draw` blurred by `std_deviation`
/// and the same blur of an exact decimation of `draw` to the grid of
/// `2^levels` pixels the blur reads its source in. `draw` paints only inside
/// `-MARGIN..VIEW + MARGIN`.
fn reduced_error(std_deviation: f32, levels: u32, mode: RenderMode, draw: &Draw) -> u8 {
    let scale = 1_u16 << levels;
    let extent = MARGIN.div_ceil(scale) * scale;
    let size = VIEW + 2 * extent;
    let offset = f64::from(extent);
    let full = render(size, size, mode, |ctx| {
        draw(ctx, Affine::translate((offset, offset)));
    });
    let decimated = decimate(&full, scale);
    let exact = paint(&decimated, sampler(ImageQuality::Low));
    let reduced = render(VIEW, VIEW, mode, |ctx| {
        ctx.push_filter_layer(blur(std_deviation));
        draw(ctx, Affine::IDENTITY);
        ctx.pop_layer();
    });
    let reference = render(VIEW, VIEW, mode, |ctx| {
        ctx.push_filter_layer(blur(std_deviation));
        ctx.set_transform(Affine::translate((-offset, -offset)) * Affine::scale(f64::from(scale)));
        ctx.set_paint(exact);
        ctx.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(decimated.width()),
            f64::from(decimated.height()),
        ));
        ctx.pop_layer();
    });
    largest_difference(&reduced, &reference)
}

#[test]
fn coarse_images_match_an_exact_decimation() {
    let columns = stripes(|x, _| x % 2 == 0);
    let checkers = stripes(|x, y| (x + y) % 2 == 0);
    let photo = photo();
    let tilted = Affine::rotate(0.5) * Affine::scale(0.37);
    let repeated = |extend| ImageSampler {
        x_extend: extend,
        y_extend: extend,
        ..sampler(ImageQuality::Medium)
    };
    let cases: [(&str, Draw); 8] = [
        (
            "columns low",
            draw_texture(&columns, sampler(ImageQuality::Low), Affine::IDENTITY),
        ),
        (
            "columns medium",
            draw_texture(&columns, sampler(ImageQuality::Medium), Affine::IDENTITY),
        ),
        (
            "checkers high",
            draw_texture(&checkers, sampler(ImageQuality::High), Affine::IDENTITY),
        ),
        (
            "photo low",
            draw_texture(&photo, sampler(ImageQuality::Low), Affine::IDENTITY),
        ),
        (
            "photo medium",
            draw_texture(&photo, sampler(ImageQuality::Medium), Affine::IDENTITY),
        ),
        (
            "photo tilted",
            draw_texture(&photo, sampler(ImageQuality::Medium), tilted),
        ),
        (
            "checkers repeat",
            draw_texture(&checkers, repeated(Extend::Repeat), Affine::scale(0.3)),
        ),
        (
            "photo reflect",
            draw_texture(&photo, repeated(Extend::Reflect), Affine::scale(0.3)),
        ),
    ];
    // The grid each blur reads its source in is `2^levels` pixels wide.
    for (std_deviation, levels) in [(100.0, 1), (300.0, 3)] {
        for (name, draw) in &cases {
            for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
                let error = reduced_error(std_deviation, levels, mode, draw);
                assert!(error <= 1, "σ {std_deviation}, {name}, {mode:?}: {error}");
            }
        }
    }
}
