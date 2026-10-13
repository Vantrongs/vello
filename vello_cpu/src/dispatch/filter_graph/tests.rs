// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A blur of a recorded layer has the pixels of the blur the recorder applies
//! to the whole pixmap of the layer at full resolution.

use crate::filter::gaussian_blur::apply_blur;
use crate::filter::pixmap::FilterPixmap;
use crate::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke};
use crate::peniko::{Color, Extend, Gradient, ImageQuality, ImageSampler};
use crate::{
    Image, ImageSource, Level, Pixmap, RasterizerSettings, RenderContext, RenderMode,
    RenderSettings, Resources,
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::color::AlphaColor;
use vello_common::filter::FilterData;
use vello_common::filter::PreparedFilter;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};

const VIEW: u16 = 64;

/// A texture of `size` by `size` pixels.
fn texture(size: u16, value: impl Fn(u16, u16) -> [u8; 4]) -> Pixmap {
    let mut pixmap = Pixmap::new(size, size);
    for (index, pixel) in pixmap.data_mut().iter_mut().enumerate() {
        let coordinate = |value: usize| u16::try_from(value).unwrap();
        let (x, y) = (
            coordinate(index % usize::from(size)),
            coordinate(index / usize::from(size)),
        );
        [pixel.r, pixel.g, pixel.b, pixel.a] = value(x, y);
    }
    pixmap
}

/// An opaque texture whose pixels are black or white.
fn stripes(size: u16, white: impl Fn(u16, u16) -> bool) -> Pixmap {
    texture(size, |x, y| {
        let value = if white(x, y) { 255 } else { 0 };
        [value, value, value, 255]
    })
}

/// A texture of smooth color fields with per-pixel noise, like a photo.
fn photo(size: u16) -> Pixmap {
    texture(size, |x, y| {
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

fn sampler(quality: ImageQuality) -> ImageSampler {
    ImageSampler {
        quality,
        ..ImageSampler::default()
    }
}

/// Something to draw, given the transform to draw it with.
type Draw = Box<dyn Fn(&mut RenderContext, Affine)>;

/// Fill the square reaching `radius` from the center of the view with
/// `texture`, sampled with `sampler`, centered on the view and transformed by
/// `transform` about its center.
fn draw_texture(texture: &Pixmap, sampler: ImageSampler, transform: Affine, radius: f64) -> Draw {
    let paint = Image {
        image: ImageSource::Pixmap(Arc::new(texture.clone())),
        sampler,
    };
    let half = f64::from(texture.width()) / 2.0;
    Box::new(move |ctx, base| {
        let center = f64::from(VIEW) / 2.0;
        ctx.set_transform(base);
        ctx.set_paint_transform(
            Affine::translate((center, center)) * transform * Affine::translate((-half, -half)),
        );
        ctx.set_paint(paint.clone());
        ctx.fill_rect(&Rect::from_center_size(
            (center, center),
            (2.0 * radius, 2.0 * radius),
        ));
    })
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

/// A square reaching `radius` from the center of the view, filled with `gradient`.
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

/// Return the largest difference between `draw` blurred by `std_deviation`
/// in the view and the blur the recorder applies to the whole pixmap of the
/// layer at full resolution. `draw` paints only inside `-extent..VIEW + extent`.
fn whole_blur_error(std_deviation: f32, extent: u16, mode: RenderMode, draw: &Draw) -> u8 {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    });
    // Squares in the corners of the extent fix the bounds of the content.
    let low = -f64::from(extent);
    let high = f64::from(VIEW) + f64::from(extent);
    let marked = |ctx: &mut RenderContext, base: Affine| {
        draw(ctx, base);
        ctx.set_transform(base);
        ctx.set_paint(Color::from_rgb8(9, 9, 9));
        ctx.fill_rect(&Rect::new(low, low, low + 4.0, low + 4.0));
        ctx.fill_rect(&Rect::new(high - 4.0, high - 4.0, high, high));
    };
    let blurred = render(VIEW, VIEW, mode, |ctx| {
        ctx.push_filter_layer(filter.clone());
        marked(ctx, Affine::IDENTITY);
        ctx.pop_layer();
    });
    assert!(
        blurred.data().iter().any(|pixel| pixel.a != 0),
        "the blur reaches the view"
    );
    // The pixmap covers the content widened by the filter padding, in tiles.
    let padding = FilterData::new(filter.clone(), Affine::IDENTITY).filter_padding;
    let tile = |value: f64| value.div_euclid(4.0) * 4.0;
    let (left, top) = (
        tile(low - padding.left as f64),
        tile(low - padding.top as f64),
    );
    let (right, bottom) = (
        -tile(-(high + padding.right as f64)),
        -tile(-(high + padding.bottom as f64)),
    );
    let side = |length: f64| u16::try_from(length as i64).unwrap();
    let (width, height) = (side(right - left), side(bottom - top));
    let source = render(width, height, mode, |ctx| {
        marked(ctx, Affine::translate((-left, -top)));
    });
    let mut pixmap = FilterPixmap::new(u32::from(width), u32::from(height));
    pixmap.data_mut().copy_from_slice(source.data());
    let mut scratch = FilterPixmap::new(u32::from(width), u32::from(height));
    let PreparedFilter::GaussianBlur(blur) = PreparedFilter::new(&filter, &Affine::IDENTITY) else {
        unreachable!("a blur prepares a blur");
    };
    apply_blur(
        &mut pixmap,
        &mut scratch,
        blur.n_decimations,
        &blur.kernel[..usize::from(blur.kernel_size)],
        blur.edge_mode,
    );
    let mut reference = Pixmap::new(VIEW, VIEW);
    let (x0, y0) = (-left as u32, -top as u32);
    for (index, pixel) in reference.data_mut().iter_mut().enumerate() {
        let (x, y) = (
            index as u32 % u32::from(VIEW),
            index as u32 / u32::from(VIEW),
        );
        *pixel = pixmap.sample(x0 + x, y0 + y);
    }
    largest_difference(&blurred, &reference)
}

#[test]
fn wide_blurs_match_the_blur_of_the_whole_pixmap() {
    // Unoptimized builds take minutes for the wider blur, whose source is
    // nine times as large, so only optimized builds check it.
    let deviations: &[u16] = if cfg!(debug_assertions) {
        &[100]
    } else {
        &[100, 300]
    };
    let center = f64::from(VIEW) / 2.0;
    let black_white = [Color::BLACK, Color::WHITE];
    let inner_blur = |std_deviation| FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    };
    let tilted = Affine::rotate(0.5) * Affine::scale(0.37);
    let repeated = |extend| ImageSampler {
        x_extend: extend,
        y_extend: extend,
        ..sampler(ImageQuality::Medium)
    };
    for &std_deviation in deviations {
        // The source covers all the blur reads, and with the pixmaps of the
        // filters nested in it, stays inside the extent.
        let radius = 3.0 * f64::from(std_deviation) + 100.0;
        let extent = 3 * std_deviation + 600;
        let size = u16::try_from(2 * radius as i64 + 8).unwrap();
        let columns = stripes(size, |x, _| x % 2 == 0);
        let checkers = stripes(size, |x, y| (x + y) % 2 == 0);
        // A quarter of the pixels lit, in a square and in a period of four,
        // at full intensity and at the least, where rounding shows.
        let corner = |x: u16, y: u16| x % 2 == 1 && y % 2 == 1;
        let fourth = |x: u16, _: u16| x % 4 == 3;
        let dim = |lit: fn(u16, u16) -> bool| {
            texture(size, move |x, y| {
                let value = u8::from(lit(x, y));
                [value, value, value, 255]
            })
        };
        let (dim_corner, dim_fourth) = (dim(corner), dim(fourth));
        let (corner, fourth) = (stripes(size, corner), stripes(size, fourth));
        let photo = photo(size);
        let cases: Vec<(&str, Draw)> = vec![
            (
                "columns low",
                draw_texture(
                    &columns,
                    sampler(ImageQuality::Low),
                    Affine::IDENTITY,
                    radius,
                ),
            ),
            (
                "checkers high",
                draw_texture(
                    &checkers,
                    sampler(ImageQuality::High),
                    Affine::IDENTITY,
                    radius,
                ),
            ),
            (
                "one of two by two",
                draw_texture(
                    &corner,
                    sampler(ImageQuality::Low),
                    Affine::IDENTITY,
                    radius,
                ),
            ),
            (
                "one of four",
                draw_texture(
                    &fourth,
                    sampler(ImageQuality::Low),
                    Affine::IDENTITY,
                    radius,
                ),
            ),
            (
                "dim one of two by two",
                draw_texture(
                    &dim_corner,
                    sampler(ImageQuality::Low),
                    Affine::IDENTITY,
                    radius,
                ),
            ),
            (
                "dim one of four",
                draw_texture(
                    &dim_fourth,
                    sampler(ImageQuality::Low),
                    Affine::IDENTITY,
                    radius,
                ),
            ),
            (
                "photo tilted",
                draw_texture(&photo, sampler(ImageQuality::Medium), tilted, radius),
            ),
            // The reference rasterizes the source in a frame moved by the
            // origin of the pixmap, where sample positions can round
            // differently in `f32`: at a scale of 0.3, this photo differs by
            // 1, so it is drawn at a power of two.
            (
                "photo reflect",
                draw_texture(
                    &photo,
                    repeated(Extend::Reflect),
                    Affine::scale(0.25),
                    radius,
                ),
            ),
            ("abutting rings", rings(1.0, radius)),
            ("lines", lines(1.0, 0.0, radius)),
            ("tilted thin lines", lines(0.5, 0.3, radius)),
            (
                "linear gradient repeating every 2 pixels",
                gradient(
                    Gradient::new_linear((center, center), (center + 2.0, center))
                        .with_extend(Extend::Repeat)
                        .with_stops(black_white),
                    radius,
                ),
            ),
            (
                "linear gradient reflected every 1.5 pixels",
                gradient(
                    Gradient::new_linear((center, center), (center + 1.5, center))
                        .with_extend(Extend::Reflect)
                        .with_stops(black_white),
                    radius,
                ),
            ),
            (
                "rings in a decimating blur",
                nested(vec![inner_blur(20.0)], rings(1.0, radius - 100.0)),
            ),
            (
                "rings in a wide blur",
                nested(vec![inner_blur(150.0)], rings(1.0, radius - 100.0)),
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
            ),
        ];
        for (name, draw) in &cases {
            for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
                let error = whole_blur_error(f32::from(std_deviation), extent, mode, draw);
                assert!(error == 0, "σ {std_deviation}, {name}, {mode:?}: {error}");
            }
        }
    }
}

/// Return the regions each node of a graph keeps, from the outermost, for
/// squares shadowed `depth` times by drop shadows that each keep their source
/// and move it by `offset(level)` pixels, the outermost by `offset(depth - 1)`.
/// The first square lies where the copy moved by `moved(depth)` lands at 16
/// in a 64 by 64 view, the others where each of `far` says.
fn shadow_regions(
    depth: u32,
    offset: impl Fn(u32) -> f32,
    moved: impl Fn(u32) -> f64,
    far: &[(f64, f64)],
) -> Vec<usize> {
    use super::super::Dispatcher;
    use super::Recording;
    use crate::filter::placed::Bounds;
    use crate::kurbo::{Affine, Rect};
    use crate::peniko::{BlendMode, Fill};
    use vello_common::paint::{Paint, PremulColor};

    let shadow = |level: u32| {
        FilterData::new(
            Filter::from_primitive(FilterPrimitive::DropShadow {
                dx: offset(level),
                dy: 0.0,
                std_deviation: 0.0,
                color: AlphaColor::from_rgba8(0, 0, 255, 255),
                edge_mode: EdgeMode::None,
            }),
            Affine::IDENTITY,
        )
    };
    let mut recording = Recording::new(
        shadow(depth - 1),
        Bounds::of_viewport(VIEW, VIEW),
        Vec::new(),
        false,
    );
    for level in (0..depth - 1).rev() {
        recording.push_layer(
            None,
            Fill::NonZero,
            Affine::IDENTITY,
            BlendMode::default(),
            1.0,
            None,
            None,
            Some(shadow(level)),
        );
    }
    let paint = || {
        Paint::Solid(PremulColor::from_alpha_color(AlphaColor::from_rgba8(
            0, 0, 255, 255,
        )))
    };
    let x = 16.0 - moved(depth);
    for (x0, x1) in core::iter::once((x, x + 1.0)).chain(far.iter().copied()) {
        recording.fill_rect_fast(
            &Rect::new(x0, 16.0, x1, 17.0),
            paint(),
            BlendMode::default(),
            None,
        );
    }
    for _ in 0..depth - 1 {
        recording.pop_layer();
    }
    let graph = recording.finish(Level::baseline());
    graph
        .nodes
        .iter()
        .map(|node| {
            [
                &node.needed,
                &node.source,
                &node.blurred,
                &node.filtered,
                &node.content,
            ]
            .iter()
            .map(|regions| regions.iter().count())
            .sum::<usize>()
        })
        .collect()
}

/// A distinct power of two times 128 pixels for each level, alternately
/// right and left.
fn alternating(level: u32) -> f32 {
    let distance = 128.0 * 2_f32.powi(i32::try_from(level).unwrap());
    if level.is_multiple_of(2) {
        distance
    } else {
        -distance
    }
}

#[test]
fn nested_shadows_keep_regions_only_for_copies_the_view_depends_on() {
    // The square every shadow moves lands in the view; the content has
    // `2^depth` copies.
    let moved = |depth: u32| (0..depth).map(|level| f64::from(alternating(level))).sum();
    for depth in [20, 24] {
        // The outermost shadow moves furthest, or least.
        let reversed = |level: u32| alternating(depth - 1 - level);
        let moved_reversed = |depth: u32| (0..depth).map(|level| f64::from(reversed(level))).sum();
        for regions in [
            shadow_regions(depth, alternating, moved, &[]),
            shadow_regions(depth, reversed, moved_reversed, &[]),
        ] {
            // Each layer keeps a few regions, however many copies its
            // content has.
            let total: usize = regions.iter().sum();
            assert!(
                total <= 8 * depth as usize,
                "{depth} layers keep {total} regions: {regions:?}"
            );
        }
    }
}

#[test]
fn shadows_keep_no_regions_for_copies_between_far_apart_squares() {
    // Every shadow moves right, so of the copies only the unmoved first
    // square lands in the view. The second square lies left of every copy of
    // the first, which the view depends on through the gap between them.
    let right = |level: u32| 128.0 * 2_f32.powi(i32::try_from(level).unwrap());
    for depth in [8, 12, 16, 20] {
        let far = -128.0 * 2_f64.powi(i32::try_from(depth).unwrap());
        let regions = shadow_regions(depth, right, |_| 0.0, &[(far, far + 32.0)]);
        let total: usize = regions.iter().sum();
        assert!(
            total <= 8 * depth as usize,
            "{depth} layers keep {total} regions: {regions:?}"
        );
    }
}

/// Return the pixels of a 64 by 64 view of squares, each given by its left
/// and top edges and its side, in nested drop shadows of the same color
/// that move them by `offsets`, from the outermost, and keep them, by
/// painting each copy.
fn shadowed_squares(squares: &[(f64, f64, f64)], offsets: &[(f64, f64)]) -> Vec<bool> {
    let mut painted = vec![false; usize::from(VIEW) * usize::from(VIEW)];
    for &(x, y, side) in squares {
        for copy in 0..1_usize << offsets.len() {
            let (mut x, mut y) = (x, y);
            for (level, (dx, dy)) in offsets.iter().enumerate() {
                if copy >> level & 1 == 1 {
                    (x, y) = (x + dx, y + dy);
                }
            }
            for row in 0..VIEW {
                for column in 0..VIEW {
                    let (px, py) = (f64::from(column), f64::from(row));
                    if (x..x + side).contains(&px) && (y..y + side).contains(&py) {
                        painted[usize::from(row) * usize::from(VIEW) + usize::from(column)] = true;
                    }
                }
            }
        }
    }
    painted
}

#[test]
fn nested_shadows_of_separate_squares_paint_every_copy_in_the_view() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |count: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % count
    };
    // Distances of each magnitude, so that copies fall in the gaps between
    // others, either way.
    let distance = |next: &mut dyn FnMut(u64) -> u64| {
        let distance = (1_i64 << (2 + next(20))) + next(64) as i64 - 32;
        (if next(2) == 0 { distance } else { -distance }) as f64
    };
    let color = AlphaColor::from_rgba8(0, 0, 255, 255);
    for case in 0..300 {
        let depth = 1 + next(6) as usize;
        let offsets: Vec<(f64, f64)> = (0..depth)
            .map(|_| {
                if next(2) == 0 {
                    (distance(&mut next), 0.0)
                } else {
                    (distance(&mut next), distance(&mut next))
                }
            })
            .collect();
        // Squares a copy of which lands in the view, and others anywhere.
        let squares: Vec<(f64, f64, f64)> = (0..1 + next(3))
            .map(|_| {
                let side = 1.0 + next(24) as f64;
                let (mut x, mut y) = (next(80) as f64 - 8.0, next(80) as f64 - 8.0);
                if next(3) == 0 {
                    (x, y) = (x + distance(&mut next), y + distance(&mut next));
                } else {
                    for (dx, dy) in &offsets {
                        if next(2) == 1 {
                            (x, y) = (x - dx, y - dy);
                        }
                    }
                }
                (x, y, side)
            })
            .collect();
        let expected = shadowed_squares(&squares, &offsets);
        for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
            let image = render(VIEW, VIEW, mode, |ctx| {
                for &(dx, dy) in &offsets {
                    #[expect(clippy::cast_possible_truncation, reason = "the distances fit `f32`")]
                    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadow {
                        dx: dx as f32,
                        dy: dy as f32,
                        std_deviation: 0.0,
                        color,
                        edge_mode: EdgeMode::None,
                    }));
                }
                ctx.set_paint(color);
                for &(x, y, side) in &squares {
                    ctx.fill_rect(&Rect::new(x, y, x + side, y + side));
                }
                for _ in &offsets {
                    ctx.pop_layer();
                }
            });
            let painted: Vec<bool> = image.data().iter().map(|pixel| pixel.a != 0).collect();
            assert!(
                painted == expected,
                "case {case}, {mode:?}: squares {squares:?}, offsets {offsets:?}"
            );
            assert!(
                image
                    .data()
                    .iter()
                    .all(|pixel| pixel.a == 0 || (pixel.b, pixel.a) == (255, 255)),
                "case {case}, {mode:?}"
            );
        }
    }
}

/// Draw the geometry of case `item`, fractional and inside the view, moved by `base`.
fn fractional(ctx: &mut RenderContext, item: usize, base: Affine) {
    ctx.set_transform(base);
    ctx.set_paint(Color::from_rgb8(30, 60, 200));
    let mut curve = BezPath::new();
    curve.move_to((5.3, 50.7));
    curve.curve_to((20.1, 2.2), (40.6, 60.3), (58.9, 9.4));
    match item {
        0 => ctx.fill_rect(&Rect::new(10.3, 10.6, 20.2, 20.9)),
        1 => ctx.fill_path(&Circle::new((30.3, 29.7), 13.1).to_path(0.1)),
        2 => {
            curve.line_to((31.25, 45.1));
            ctx.set_stroke(Stroke::new(2.3));
            ctx.stroke_path(&curve);
        }
        3 => {
            ctx.set_stroke(Stroke::new(1.7).with_dashes(0.3, [2.1, 1.3]));
            ctx.stroke_path(&curve);
        }
        _ => {
            ctx.push_clip_path(&Circle::new((33.1, 31.7), 21.3).to_path(0.1));
            ctx.fill_rect(&Rect::new(8.7, 12.2, 56.1, 47.9));
            ctx.pop_clip();
        }
    }
}

/// An offset moves the pixels the recorder draws in the viewport of the
/// layer's source, where geometry is rounded: windows of it that start inside
/// the geometry, where the offset brings it in from the right, round it there too.
#[test]
fn an_offset_moves_the_geometry_drawn_in_the_viewport_of_the_source() {
    for (dx, dy) in [(3.0, 1.0), (-13.0, 5.0), (-7.0, -9.0), (21.0, 0.0)] {
        let filter = Filter::from_primitive(FilterPrimitive::Offset { dx, dy });
        let data = FilterData::new(filter.clone(), Affine::IDENTITY);
        let (left, top) = data.source_shift();
        let size = data.source_viewport_size(VIEW, VIEW).unwrap();
        for item in 0..5 {
            for shift in [(0.0, 0.0), (0.37, 0.61), (-2.13, 1.9)] {
                for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
                    let moved = render(VIEW, VIEW, mode, |ctx| {
                        ctx.push_filter_layer(filter.clone());
                        fractional(ctx, item, Affine::translate(shift));
                        ctx.pop_layer();
                    });
                    let base =
                        Affine::translate((left as f64, top as f64)) * Affine::translate(shift);
                    let source = render(size.width(), size.height(), mode, |ctx| {
                        fractional(ctx, item, base);
                    });
                    let mut expected = Pixmap::new(VIEW, VIEW);
                    for (index, pixel) in expected.data_mut().iter_mut().enumerate() {
                        let (x, y) = (index % usize::from(VIEW), index / usize::from(VIEW));
                        let from = |at: usize, by: f32, shift: u64, side: u16| {
                            let at = at as i64 - by as i64 + shift as i64;
                            usize::try_from(at)
                                .ok()
                                .filter(|at| *at < usize::from(side))
                        };
                        if let (Some(u), Some(v)) = (
                            from(x, dx, left, size.width()),
                            from(y, dy, top, size.height()),
                        ) {
                            *pixel = source.data()[v * usize::from(size.width()) + u];
                        }
                    }
                    assert!(
                        moved.data() == expected.data(),
                        "offset ({dx}, {dy}), item {item}, shift {shift:?}, {mode:?}: {} pixels differ",
                        moved
                            .data()
                            .iter()
                            .zip(expected.data())
                            .filter(|(a, b)| a != b)
                            .count()
                    );
                }
            }
        }
    }
}

/// Curves crowded into the tiles left of a window, under a clip reaching
/// past them, change what the window draws through the rounding of their
/// coverage, so they are flattened as in the recorder's whole viewport: the
/// window culls them to the bounds of the clip's strips there.
#[test]
fn windows_flatten_curves_left_of_them_as_the_viewport_does() {
    // The offset moves the source left by this many pixels.
    let moved_by = 30;
    let filter = Filter::from_primitive(FilterPrimitive::Offset {
        dx: -(moved_by as f32),
        dy: 0.0,
    });
    let size = FilterData::new(filter.clone(), Affine::IDENTITY)
        .source_viewport_size(VIEW, VIEW)
        .unwrap();
    let clip = Circle::new((50.3, 32.1), 70.0).to_path(0.1);
    struct Random(u64);
    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn number(&mut self, low: f64, high: f64) -> f64 {
            low + (self.next() % 1_000_000) as f64 / 1_000_000.0 * (high - low)
        }
    }
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    for case in 0..1000 {
        let mut path = BezPath::new();
        let mut y = random.number(-5.0, 10.0);
        path.move_to((random.number(60.0, 90.0), y));
        path.line_to((random.number(2.0, 26.0), y));
        for _ in 0..40 {
            let y1 = y + random.number(0.2, 2.0);
            path.curve_to(
                (random.number(16.0, 27.9), random.number(y, y1)),
                (random.number(16.0, 27.9), random.number(y, y1)),
                (random.number(16.0, 27.9), y1),
            );
            y = y1;
        }
        path.line_to((random.number(60.0, 90.0), y));
        path.close_path();
        let width = (random.next() % 2 == 1).then(|| random.number(0.3, 3.0));
        let mode = [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality][case % 2];
        let draw = |ctx: &mut RenderContext| {
            ctx.push_clip_path(&clip);
            ctx.set_paint(Color::from_rgb8(30, 60, 200));
            match width {
                Some(width) => {
                    ctx.set_stroke(Stroke::new(width));
                    ctx.stroke_path(&path);
                }
                None => ctx.fill_path(&path),
            }
            ctx.pop_clip();
        };
        let moved = render(VIEW, VIEW, mode, |ctx| {
            ctx.push_filter_layer(filter.clone());
            draw(ctx);
            ctx.pop_layer();
        });
        let source = render(size.width(), size.height(), mode, draw);
        let width = usize::from(VIEW);
        let differ = moved
            .data()
            .iter()
            .enumerate()
            .filter(|(index, pixel)| {
                let (x, y) = (index % width, index / width);
                **pixel != source.data()[y * usize::from(size.width()) + x + moved_by]
            })
            .count();
        assert_eq!(differ, 0, "case {case}: {} {width:?}", path.to_svg());
    }
}

/// The recorder's viewport of a source may be up to the whole `u16` domain
/// wide, past the largest tile multiple in it, and the source is drawn there
/// as the recorder draws it: an offset bringing in its far edge shows the
/// pixels the recorder draws there.
#[test]
fn an_offset_shows_the_far_edge_of_the_widest_viewports_as_drawn_there() {
    let moved_by = 65_472;
    let filter = Filter::from_primitive(FilterPrimitive::Offset {
        dx: -(moved_by as f32),
        dy: 0.0,
    });
    let edge = f64::from(moved_by);
    for view in [61, 62, 63] {
        let viewport = view + moved_by;
        for item in 0..2 {
            let draw = |ctx: &mut RenderContext| {
                ctx.set_paint(Color::from_rgb8(30, 60, 200));
                if item == 0 {
                    ctx.fill_path(&Circle::new((edge + 55.37, 30.3), 9.13).to_path(0.1));
                } else {
                    let mut curve = BezPath::new();
                    curve.move_to((edge + 40.3, 5.7));
                    curve.curve_to(
                        (edge + 80.1, 20.2),
                        (edge + 30.6, 50.3),
                        (edge + 66.9, 59.4),
                    );
                    ctx.set_stroke(Stroke::new(1.7));
                    ctx.stroke_path(&curve);
                }
            };
            for mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
                let moved = render(view, VIEW, mode, |ctx| {
                    ctx.push_filter_layer(filter.clone());
                    draw(ctx);
                    ctx.pop_layer();
                });
                let source = render(viewport, VIEW, mode, draw);
                let (view, viewport, moved_by) = (
                    usize::from(view),
                    usize::from(viewport),
                    usize::from(moved_by),
                );
                let differ = moved
                    .data()
                    .iter()
                    .enumerate()
                    .filter(|(index, pixel)| {
                        let (x, y) = (index % view, index / view);
                        **pixel != source.data()[y * viewport + x + moved_by]
                    })
                    .count();
                assert_eq!(differ, 0, "view {view}, item {item}, {mode:?}");
            }
        }
    }
}
