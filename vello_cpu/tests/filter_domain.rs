// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter source coordinates and intermediate buffers can exceed the physical u16 target.

use std::sync::Arc;
use vello_cpu::color::palette::css::{BLUE, RED};
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, BezPath, Rect, Shape};
use vello_cpu::peniko::{ColorStop, ColorStops, Gradient, ImageSampler, LinearGradientPosition};
use vello_cpu::{
    Image, ImageSource, Level, Pixmap, RasterizerSettings, RenderContext, RenderMode,
    RenderSettings, Resources,
};

const CLEAR: [u8; 4] = [0; 4];
const RED_PIXEL: [u8; 4] = [255, 0, 0, 255];

fn modes() -> impl Iterator<Item = RenderMode> {
    [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ]
    .into_iter()
}

fn context(length: u16, vertical: bool) -> RenderContext {
    let (width, height) = dimensions(length, vertical);
    // Filters currently use the single-threaded dispatcher.
    RenderContext::new_with(
        width,
        height,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    )
}

fn dimensions(length: u16, vertical: bool) -> (u16, u16) {
    if vertical { (4, length) } else { (length, 4) }
}

fn along(major: f64, minor: f64, vertical: bool) -> (f64, f64) {
    if vertical {
        (minor, major)
    } else {
        (major, minor)
    }
}

fn rect(start: f64, end: f64, vertical: bool) -> Rect {
    let (x0, y0) = along(start, 0.0, vertical);
    let (x1, y1) = along(end, 4.0, vertical);
    Rect::new(x0, y0, x1, y1)
}

fn offset(distance: f32, vertical: bool) -> Filter {
    let (dx, dy) = if vertical {
        (0.0, distance)
    } else {
        (distance, 0.0)
    };
    Filter::from_primitive(FilterPrimitive::Offset { dx, dy })
}

fn blur() -> Filter {
    Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 8.0,
        edge_mode: EdgeMode::None,
    })
}

fn finish(mut ctx: RenderContext, length: u16, vertical: bool, mode: RenderMode) -> Pixmap {
    ctx.flush();
    let (width, height) = dimensions(length, vertical);
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

fn pixel(image: &Pixmap, major: usize, minor: usize, vertical: bool) -> [u8; 4] {
    let (x, y) = if vertical {
        (minor, major)
    } else {
        (major, minor)
    };
    let index = y * usize::from(image.width()) + x;
    image.data_as_u8_slice()[index * 4..index * 4 + 4]
        .try_into()
        .unwrap()
}

fn assert_translated(actual: &Pixmap, reference: &Pixmap, length: u16, short: u16, vertical: bool) {
    assert!(reference.data().iter().any(|p| p.a != 0), "empty reference");
    let shift = usize::from(length - short);
    for major in 0..usize::from(length) {
        for minor in 0..4 {
            let expected = if major < shift {
                CLEAR
            } else {
                pixel(reference, major - shift, minor, vertical)
            };
            assert_eq!(
                pixel(actual, major, minor, vertical),
                expected,
                "length={length}, vertical={vertical}, pixel=({major},{minor})"
            );
        }
    }
}

fn tail(length: u16, vertical: bool, mode: RenderMode, blurred: bool) -> Pixmap {
    let mut ctx = context(length, vertical);
    ctx.push_filter_layer(if blurred {
        blur()
    } else {
        offset(-32.0, vertical)
    });
    ctx.set_paint(RED);
    let end = f64::from(length) + if blurred { 0.0 } else { 32.0 };
    ctx.fill_rect(&rect(end - 4.0, end, vertical));
    ctx.pop_layer();
    finish(ctx, length, vertical, mode)
}

#[test]
fn source_halo_beyond_u16_matches_small_reference_exactly() {
    for mode in modes() {
        for length in [65_520, 65_528, 65_532, 65_535] {
            for vertical in [false, true] {
                for blurred in [false, true] {
                    // Identical tight source bounds up to whole-tile translation preserve
                    // all decimation sizes and the blur's local sampling phase.
                    let short = 64 + length % 4;
                    let reference = tail(short, vertical, mode, blurred);
                    let actual = tail(length, vertical, mode, blurred);
                    assert_translated(&actual, &reference, length, short, vertical);
                }
            }
        }
    }
}

#[test]
fn full_intermediate_buffer_exceeds_u16_in_both_orientations() {
    for mode in modes() {
        for length in [65_520, 65_528, 65_532, 65_535] {
            for vertical in [false, true] {
                for distance in [-32.0_f32, 32.0] {
                    let mut ctx = context(length, vertical);
                    ctx.push_filter_layer(offset(distance, vertical));
                    ctx.set_paint(RED);
                    // These full-width sources, unlike a four-pixel edge fixture, require
                    // one allocated intermediate dimension greater than 65535.
                    let start = -f64::from(distance.max(0.0));
                    let end = f64::from(length) - f64::from(distance.min(0.0));
                    ctx.fill_rect(&rect(start, end, vertical));
                    ctx.pop_layer();
                    let image = finish(ctx, length, vertical, mode);
                    assert!(
                        image
                            .data_as_u8_slice()
                            .chunks_exact(4)
                            .all(|p| p == RED_PIXEL),
                        "length={length}, vertical={vertical}, distance={distance}, mode={mode:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn source_coordinates_and_padding_exceed_packed_tile_domain() {
    for mode in modes() {
        for vertical in [false, true] {
            for magnitude in [65_568.0_f32, 262_148.0] {
                for distance in [-magnitude, magnitude] {
                    let mut ctx = context(12, vertical);
                    ctx.push_filter_layer(offset(distance, vertical));
                    ctx.set_paint(RED);
                    let start = 4.0 - f64::from(distance);
                    ctx.fill_rect(&rect(start, start + 4.0, vertical));
                    ctx.pop_layer();
                    let image = finish(ctx, 12, vertical, mode);
                    for major in 0..12 {
                        for minor in 0..4 {
                            assert_eq!(
                                pixel(&image, major, minor, vertical),
                                if (4..8).contains(&major) {
                                    RED_PIXEL
                                } else {
                                    CLEAR
                                },
                                "distance={distance}, vertical={vertical}, mode={mode:?}, pixel=({major},{minor})"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn nested(length: u16, vertical: bool, mode: RenderMode) -> Pixmap {
    let mut ctx = context(length, vertical);
    // A scale affects the filter offset, while the translation positions a small
    // source beyond the root viewport. A curved clip must survive both layers.
    ctx.set_transform(
        Affine::translate(along(f64::from(length) - 8.0, 0.0, vertical)) * Affine::scale(2.0),
    );
    ctx.push_filter_layer(offset(-16.0, vertical));
    ctx.push_filter_layer(blur());
    let mut clip = BezPath::new();
    clip.move_to(along(16.0, 0.0, vertical));
    clip.curve_to(
        along(20.0, 0.0, vertical),
        along(20.0, 2.0, vertical),
        along(16.0, 2.0, vertical),
    );
    clip.close_path();
    ctx.push_clip_layer(&clip);
    ctx.set_paint(RED);
    ctx.fill_rect(&rect(16.0, 20.0, vertical));
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();
    finish(ctx, length, vertical, mode)
}

#[test]
fn nested_filters_preserve_curved_clip_and_transform_beyond_u16() {
    for mode in modes() {
        for length in [65_520, 65_535] {
            for vertical in [false, true] {
                let short = 128 + length % 4;
                let reference = nested(short, vertical, mode);
                let actual = nested(length, vertical, mode);
                assert_translated(&actual, &reference, length, short, vertical);
            }
        }
    }
}

#[test]
fn invalid_filter_is_rejected_before_mutating_existing_layer() {
    for mode in modes() {
        for (length, filter, reason) in [
            (
                12,
                offset(f32::MAX, false),
                "filter padding exceeds u32 coordinate domain",
            ),
            (
                12,
                Filter::from_primitive(FilterPrimitive::GaussianBlur {
                    std_deviation: f32::INFINITY,
                    edge_mode: EdgeMode::None,
                }),
                "filter standard deviation must be finite and non-negative",
            ),
            (
                512,
                offset(4_294_967_040.0, false),
                "filter source viewport exceeds u32 coordinate domain",
            ),
        ] {
            let mut ctx = context(length, false);
            ctx.push_opacity_layer(0.5);
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ctx.push_filter_layer(filter)
            }))
            .expect_err("unrepresentable source domain must be rejected");
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied());
            assert_eq!(message, Some(reason));
            ctx.set_paint(RED);
            ctx.fill_rect(&rect(0.0, f64::from(length), false));
            ctx.pop_layer();
            let image = finish(ctx, length, false, mode);
            assert!(
                image
                    .data_as_u8_slice()
                    .chunks_exact(4)
                    .all(|p| p == [128, 0, 0, 128])
            );
        }
    }
}

fn full_blur(length: u16, vertical: bool, mode: RenderMode) -> Pixmap {
    let mut ctx = context(length, vertical);
    ctx.push_filter_layer(blur());
    ctx.set_paint(RED);
    ctx.fill_rect(&rect(0.0, f64::from(length), vertical));
    ctx.pop_layer();
    finish(ctx, length, vertical, mode)
}

#[test]
fn full_wide_blur_preserves_both_edges_and_constant_interior() {
    for mode in modes() {
        for length in [65_520, 65_528, 65_532, 65_535] {
            for vertical in [false, true] {
                // Sigma 8 uses two decimation levels. Equal lengths modulo four
                // preserve the right-edge reconstruction phase; 128 pixels exceed
                // the finite kernel support, separating the two edges completely.
                let short = 256 + length % 4;
                let reference = full_blur(short, vertical, mode);
                let actual = full_blur(length, vertical, mode);
                assert!(reference.data().iter().any(|p| p.a != 0));
                for major in 0..usize::from(length) {
                    let reference_major = if major < 128 {
                        major
                    } else if major >= usize::from(length) - 128 {
                        major - usize::from(length - short)
                    } else {
                        128
                    };
                    for minor in 0..4 {
                        assert_eq!(
                            pixel(&actual, major, minor, vertical),
                            pixel(&reference, reference_major, minor, vertical),
                            "length={length}, vertical={vertical}, mode={mode:?}, pixel=({major},{minor})"
                        );
                    }
                }
            }
        }
    }
}

fn shadow_tail(length: u16, vertical: bool, mode: RenderMode) -> Pixmap {
    let mut ctx = context(length, vertical);
    let (dx, dy) = if vertical { (0.0, -32.0) } else { (-32.0, 0.0) };
    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadowOnly {
        dx,
        dy,
        std_deviation: 1.5,
        color: RED,
        edge_mode: EdgeMode::None,
    }));
    ctx.set_paint(RED);
    ctx.fill_rect(&rect(
        f64::from(length) + 28.0,
        f64::from(length) + 32.0,
        vertical,
    ));
    ctx.pop_layer();
    finish(ctx, length, vertical, mode)
}

#[test]
fn offscreen_shadow_source_is_retained_beyond_u16() {
    for mode in modes() {
        for length in [65_520, 65_528, 65_532, 65_535] {
            for vertical in [false, true] {
                let short = 64 + length % 4;
                let reference = shadow_tail(short, vertical, mode);
                let actual = shadow_tail(length, vertical, mode);
                assert_translated(&actual, &reference, length, short, vertical);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PrecisionScene {
    Rect,
    Path,
    CurvedClip,
    Image,
    Gradient,
}

fn precision_scene(shift: f32, vertical: bool, mode: RenderMode, scene: PrecisionScene) -> Pixmap {
    let mut ctx = context(12, vertical);
    ctx.set_transform(Affine::translate(along(f64::from(shift), 0.0, vertical)));
    ctx.push_filter_layer(offset(-shift, vertical));
    ctx.set_paint(RED);
    let fractional = rect(1.0 / 256.0, 1.0, vertical);
    match scene {
        PrecisionScene::Rect => ctx.fill_rect(&fractional),
        PrecisionScene::Path => {
            let mut path = BezPath::new();
            path.move_to(along(1.0 / 256.0, 0.0, vertical));
            path.line_to(along(2.5, 0.0, vertical));
            path.line_to(along(1.75, 4.0, vertical));
            path.line_to(along(1.0 / 256.0, 4.0, vertical));
            path.close_path();
            ctx.fill_path(&path);
        }
        PrecisionScene::CurvedClip => {
            let mut path = BezPath::new();
            path.move_to(along(1.0 / 256.0, 0.0, vertical));
            path.curve_to(
                along(3.25, 0.0, vertical),
                along(3.25, 4.0, vertical),
                along(1.0 / 256.0, 4.0, vertical),
            );
            path.close_path();
            ctx.push_clip_layer(&path);
            ctx.fill_rect(&rect(0.0, 4.0, vertical));
            ctx.pop_layer();
        }
        PrecisionScene::Image => {
            let mut source = Pixmap::new(4, 4);
            for (index, pixel) in source.data_mut().iter_mut().enumerate() {
                *pixel = if index % 3 == 0 { RED } else { BLUE }
                    .premultiply()
                    .to_rgba8();
            }
            ctx.set_paint(Image {
                image: ImageSource::Pixmap(Arc::new(source)),
                sampler: ImageSampler::default(),
            });
            ctx.fill_path(&rect(1.0 / 256.0, 4.0, vertical).to_path(0.1));
        }
        PrecisionScene::Gradient => {
            ctx.set_paint(Gradient {
                kind: LinearGradientPosition {
                    start: along(0.0, 0.0, vertical).into(),
                    end: along(4.0, 0.0, vertical).into(),
                }
                .into(),
                stops: ColorStops::from(
                    [ColorStop::from((0.0, RED)), ColorStop::from((1.0, BLUE))].as_slice(),
                ),
                ..Default::default()
            });
            ctx.fill_path(&rect(1.0 / 256.0, 4.0, vertical).to_path(0.1));
        }
    }
    ctx.pop_layer();
    finish(ctx, 12, vertical, mode)
}

#[test]
fn wide_source_preserves_subpixel_geometry_and_paint_precision() {
    let mut failures = Vec::new();
    for mode in modes() {
        for vertical in [false, true] {
            for scene in [
                PrecisionScene::Rect,
                PrecisionScene::Path,
                PrecisionScene::CurvedClip,
                PrecisionScene::Image,
                PrecisionScene::Gradient,
            ] {
                let reference = precision_scene(0.0, vertical, mode, scene);
                assert!(
                    reference.data().iter().any(|p| p.a != 0),
                    "empty local reference {scene:?}"
                );
                if matches!(scene, PrecisionScene::Rect) {
                    assert_eq!(pixel(&reference, 0, 0, vertical), [254, 0, 0, 254]);
                }
                for shift in [70_000.0, 16_777_216.0] {
                    let actual = precision_scene(shift, vertical, mode, scene);
                    if let Some((index, (actual, expected))) = actual
                        .data_as_u8_slice()
                        .chunks_exact(4)
                        .zip(reference.data_as_u8_slice().chunks_exact(4))
                        .enumerate()
                        .find(|(_, (a, e))| a != e)
                    {
                        failures.push(format!("{scene:?}, shift={shift}, vertical={vertical}, mode={mode:?}, pixel={index}, actual={actual:?}, expected={expected:?}"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} precision failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn pure_offset_rounding_and_destination_crop_remain_exact() {
    for mode in modes() {
        for vertical in [false, true] {
            for distance in [
                -70_000.5_f32,
                -65_536.5,
                -3.5,
                -0.5,
                0.5,
                3.5,
                65_536.5,
                70_000.5,
            ] {
                let mut ctx = context(12, vertical);
                ctx.push_filter_layer(offset(distance, vertical));
                ctx.set_paint(RED);
                // Offset rounds half away from zero. This three-pixel source maps
                // to [3,6), whose two edges both cut through destination tiles.
                let start = 3.0 - f64::from(distance.round());
                ctx.fill_rect(&rect(start, start + 3.0, vertical));
                ctx.pop_layer();
                let image = finish(ctx, 12, vertical, mode);
                for major in 0..12 {
                    for minor in 0..4 {
                        assert_eq!(
                            pixel(&image, major, minor, vertical),
                            if (3..6).contains(&major) {
                                RED_PIXEL
                            } else {
                                CLEAR
                            },
                            "distance={distance}, vertical={vertical}, mode={mode:?}, pixel=({major},{minor})"
                        );
                    }
                }
            }
        }
    }
}
