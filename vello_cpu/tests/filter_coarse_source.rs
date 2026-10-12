// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A blur wide enough to read its source at a coarser resolution still sees
//! every pixel of that source.

use std::sync::Arc;
use vello_cpu::color::AlphaColor;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Circle, Point, Rect, Shape, Stroke};
use vello_cpu::peniko::{Color, Extend, Gradient, ImageQuality, ImageSampler};
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
/// `-extent..VIEW + extent`.
fn reduced_error(
    std_deviation: f32,
    levels: u32,
    extent: u16,
    mode: RenderMode,
    draw: &Draw,
) -> u8 {
    let scale = 1_u16 << levels;
    // Moving the source by a multiple of the tiles of every grid a blur in it
    // reads leaves where those blurs anchor their decimations.
    let extent = extent.div_ceil(512) * 512;
    let size = VIEW + 2 * extent;
    let offset = f64::from(extent);
    // Squares in the corners of the extent give the source and the decimated
    // reference the same bounds, which anchor the decimations of their blurs.
    let marked = |ctx: &mut RenderContext, base: Affine| {
        draw(ctx, base);
        ctx.set_transform(base);
        ctx.set_paint(Color::from_rgb8(9, 9, 9));
        let (low, high, side) = (-offset, f64::from(VIEW) + offset, f64::from(scale));
        ctx.fill_rect(&Rect::new(low, low, low + side, low + side));
        ctx.fill_rect(&Rect::new(high - side, high - side, high, high));
    };
    let full = render(size, size, mode, |ctx| {
        marked(ctx, Affine::translate((offset, offset)));
    });
    let decimated = decimate(&full, scale);
    let exact = paint(&decimated, sampler(ImageQuality::Low));
    let reduced = render(VIEW, VIEW, mode, |ctx| {
        ctx.push_filter_layer(blur(std_deviation));
        marked(ctx, Affine::IDENTITY);
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
    assert!(
        reduced.data().iter().any(|pixel| pixel.a != 0),
        "the blur reaches the view"
    );
    largest_difference(&reduced, &reference)
}

/// Concentric rings one pixel wide every `spacing` pixels out to `radius`,
/// alternating between two colors.
fn rings(spacing: f64, radius: f64) -> Draw {
    Box::new(move |ctx, base| {
        ctx.set_transform(base);
        ctx.set_stroke(Stroke::new(1.0));
        let center = f64::from(VIEW) / 2.0;
        let rings = (1..).take_while(|&ring| f64::from(ring) * spacing <= radius);
        for ring in rings {
            ctx.set_paint(if ring % 2 == 0 {
                Color::from_rgb8(230, 40, 20)
            } else {
                Color::from_rgb8(20, 60, 240)
            });
            let circle = Circle::new((center, center), f64::from(ring) * spacing);
            ctx.stroke_path(&circle.to_path(0.1));
        }
    })
}

/// Lines `width` pixels wide every two pixels, rotated by `angle`, inside a
/// square reaching `radius` from the center of the view.
fn lines(width: f64, angle: f64, radius: f64) -> Draw {
    Box::new(move |ctx, base| {
        let center = f64::from(VIEW) / 2.0;
        ctx.set_transform(base);
        ctx.push_clip_layer(
            &Rect::from_center_size((center, center), (2.0 * radius, 2.0 * radius)).to_path(0.1),
        );
        ctx.set_transform(base * Affine::rotate_about(angle, Point::new(center, center)));
        ctx.set_paint(Color::BLACK);
        let reach = 1.5 * radius;
        let lines = (0..).map(|line| 2.0 * f64::from(line) - reach);
        for x in lines.take_while(|&x| x < reach) {
            let x = center + x;
            ctx.fill_rect(&Rect::new(x, center - reach, x + width, center + reach));
        }
        ctx.pop_layer();
    })
}

/// A rectangle reaching `radius` from the center of the view, filled with `gradient`.
fn gradient(gradient: Gradient, radius: f64) -> Draw {
    Box::new(move |ctx, base| {
        let center = f64::from(VIEW) / 2.0;
        ctx.set_transform(base);
        ctx.set_paint(gradient.clone());
        ctx.fill_rect(&Rect::from_center_size(
            (center, center),
            (2.0 * radius, 2.0 * radius),
        ));
    })
}

/// `draw` nested in `filters`, each in the one before.
fn nested(filters: Vec<FilterPrimitive>, draw: Draw) -> Draw {
    Box::new(move |ctx, base| {
        ctx.set_transform(base);
        for filter in &filters {
            ctx.push_filter_layer(Filter::from_primitive(filter.clone()));
        }
        draw(ctx, base);
        for _ in &filters {
            ctx.pop_layer();
        }
    })
}

#[test]
fn wide_blurs_match_a_blur_of_an_exact_decimation() {
    let columns = stripes(|x, _| x % 2 == 0);
    let checkers = stripes(|x, y| (x + y) % 2 == 0);
    let photo = photo();
    let tilted = Affine::rotate(0.5) * Affine::scale(0.37);
    let repeated = |extend| ImageSampler {
        x_extend: extend,
        y_extend: extend,
        ..sampler(ImageQuality::Medium)
    };
    let center = f64::from(VIEW) / 2.0;
    let black_white = [Color::BLACK, Color::WHITE];
    let inner_blur = |std_deviation| FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    };
    // The grid each blur reads its source in is `2^levels` pixels wide.
    // Unoptimized builds take minutes for the wider blur, whose source is
    // nine times as large, so only optimized builds check it.
    let deviations: &[(u16, u32)] = if cfg!(debug_assertions) {
        &[(100, 1)]
    } else {
        &[(100, 1), (300, 3)]
    };
    for &(std_deviation, levels) in deviations {
        // The source covers all the blur reads, and with what the filters
        // nested in it add, stays inside the extent.
        let radius = 3.0 * f64::from(std_deviation) + 100.0;
        let extent = 3 * std_deviation + 300;
        let cases: Vec<(&str, Draw, bool)> = vec![
            (
                "columns low",
                draw_texture(&columns, sampler(ImageQuality::Low), Affine::IDENTITY),
                false,
            ),
            (
                "checkers high",
                draw_texture(&checkers, sampler(ImageQuality::High), Affine::IDENTITY),
                false,
            ),
            (
                "photo tilted",
                draw_texture(&photo, sampler(ImageQuality::Medium), tilted),
                false,
            ),
            (
                "photo reflect",
                draw_texture(&photo, repeated(Extend::Reflect), Affine::scale(0.3)),
                false,
            ),
            ("abutting rings", rings(1.0, radius), false),
            ("lines", lines(1.0, 0.0, radius), false),
            ("tilted thin lines", lines(0.5, 0.3, radius), false),
            (
                "linear gradient repeating every 2 pixels",
                gradient(
                    Gradient::new_linear((center, center), (center + 2.0, center))
                        .with_extend(Extend::Repeat)
                        .with_stops(black_white),
                    radius,
                ),
                false,
            ),
            (
                "linear gradient reflected every 1.5 pixels",
                gradient(
                    Gradient::new_linear((center, center), (center + 1.5, center))
                        .with_extend(Extend::Reflect)
                        .with_stops(black_white),
                    radius,
                ),
                false,
            ),
            (
                "rings in a decimating blur",
                nested(vec![inner_blur(20.0)], rings(1.0, radius - 100.0)),
                true,
            ),
            (
                "rings in a wide blur",
                nested(vec![inner_blur(150.0)], rings(1.0, radius - 100.0)),
                true,
            ),
            (
                "lines in an offset and a drop shadow",
                nested(
                    vec![
                        FilterPrimitive::Offset { dx: 13.0, dy: -7.0 },
                        FilterPrimitive::DropShadow {
                            dx: 37.0,
                            dy: -11.0,
                            std_deviation: 5.0,
                            color: AlphaColor::from_rgba8(200, 30, 60, 200),
                            edge_mode: EdgeMode::None,
                        },
                    ],
                    lines(1.0, 0.3, radius - 100.0),
                ),
                true,
            ),
        ];
        for (name, draw, nests_filters) in &cases {
            for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
                let error = reduced_error(f32::from(std_deviation), levels, extent, mode, draw);
                // Rasterizing at higher precision rounds coverage differently
                // where the source is placed differently, by up to 1 after
                // the nested filters, which the reference shows too when it
                // is moved by a multiple of every tile.
                let tolerance = u8::from(*nests_filters && mode == RenderMode::OptimizeQuality);
                assert!(
                    error <= tolerance,
                    "σ {std_deviation}, {name}, {mode:?}: {error}"
                );
            }
        }
    }
}
