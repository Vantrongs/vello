// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact pixels at the last representable frame dimensions, in both orientations.

use std::sync::Arc;
use vello_cpu::color::palette::css::{BLUE, RED};
use vello_cpu::kurbo::{Affine, Rect, Shape};
use vello_cpu::peniko::{ColorStop, ColorStops, Gradient, ImageSampler, LinearGradientPosition};
use vello_cpu::{
    Image, ImageSource, Level, Mask, Pixmap, RasterizerSettings, RenderContext, RenderMode,
    RenderSettings, Resources, TargetInit,
};

const CLEAR: [u8; 4] = [0, 0, 0, 0];
const RED_PIXEL: [u8; 4] = [255, 0, 0, 255];
const BLUE_PIXEL: [u8; 4] = [0, 0, 255, 255];

fn modes() -> impl Iterator<Item = RenderMode> {
    [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ]
    .into_iter()
}

fn threads() -> impl Iterator<Item = u16> {
    [
        0,
        #[cfg(feature = "multithreading")]
        2,
    ]
    .into_iter()
}

fn context(width: u16, height: u16, num_threads: u16, level: Level) -> RenderContext {
    RenderContext::new_with(width, height, RenderSettings { level, num_threads })
}

fn rect(start: u16, end: u16, thin: u16, transpose: bool) -> Rect {
    let (start, end, thin) = (f64::from(start), f64::from(end), f64::from(thin));
    if transpose {
        Rect::new(0.0, start, thin, end)
    } else {
        Rect::new(start, 0.0, end, thin)
    }
}

fn check_pixels(image: &Pixmap, label: &str, expected: impl Fn(usize, usize) -> [u8; 4]) {
    let width = usize::from(image.width());
    for (i, actual) in image
        .data_as_u8_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
    {
        let (x, y) = (i % width, i / width);
        assert_eq!(*actual, expected(x, y), "{label}: pixel ({x}, {y})");
    }
}

#[derive(Clone, Copy, Debug)]
enum Scene {
    Rect,
    Path,
    TranslatedTail,
    NestedClip,
    Opacity,
    DepthOverlap,
    PaintMask,
    LayerMask,
    Gradient,
    Image,
    Empty,
    SrcOver,
    SrcOverTail,
}

fn boundary(length: u16) {
    boundary_with(
        length,
        Level::baseline(),
        &[1, 4],
        &[
            Scene::Rect,
            Scene::Path,
            Scene::TranslatedTail,
            Scene::NestedClip,
            Scene::Opacity,
            Scene::DepthOverlap,
            Scene::PaintMask,
            Scene::LayerMask,
            Scene::Gradient,
            Scene::Image,
            Scene::Empty,
            Scene::SrcOver,
            Scene::SrcOverTail,
        ],
    );
}

fn boundary_with(length: u16, level: Level, thicknesses: &[u16], scenes: &[Scene]) {
    for num_threads in threads() {
        for render_mode in modes() {
            for &thin in thicknesses {
                for transpose in [false, true] {
                    let (width, height) = if transpose {
                        (thin, length)
                    } else {
                        (length, thin)
                    };
                    let mut ctx = context(width, height, num_threads, level);
                    let mut resources = Resources::new();
                    let mut image = Pixmap::new(width, height);
                    let full = rect(0, length, thin, transpose);
                    let tail = rect(length - 3, length, thin, transpose);
                    for &scene in scenes {
                        ctx.reset();
                        // Clearing and SrcOver must both work on an already populated target.
                        image.data_mut().fill(BLUE.premultiply().to_rgba8());
                        ctx.set_paint(RED);
                        match scene {
                            Scene::Rect => ctx.fill_rect(&full),
                            Scene::SrcOverTail => ctx.fill_rect(&tail),
                            Scene::Path => ctx.fill_path(&full.to_path(0.1)),
                            Scene::TranslatedTail => {
                                let shift = f64::from(length - 3);
                                ctx.set_transform(Affine::translate(if transpose {
                                    (0.0, shift)
                                } else {
                                    (shift, 0.0)
                                }));
                                ctx.fill_path(&rect(0, 3, thin, transpose).to_path(0.1));
                            }
                            Scene::NestedClip => {
                                ctx.push_clip_layer(&full.to_path(0.1));
                                ctx.push_clip_layer(&tail.to_path(0.1));
                                ctx.fill_path(&full.to_path(0.1));
                                ctx.pop_layer();
                                ctx.pop_layer();
                            }
                            Scene::Opacity => {
                                ctx.push_opacity_layer(0.5);
                                ctx.fill_rect(&full);
                                ctx.pop_layer();
                            }
                            Scene::DepthOverlap => {
                                ctx.fill_rect(&full);
                                ctx.set_paint(BLUE);
                                ctx.fill_rect(&rect(0, length - 3, thin, transpose));
                            }
                            Scene::PaintMask | Scene::LayerMask => {
                                let data = (0..usize::from(width) * usize::from(height))
                                    .map(|i| {
                                        let major = if transpose {
                                            i / usize::from(width)
                                        } else {
                                            i % usize::from(width)
                                        };
                                        if major >= usize::from(length - 3) {
                                            255
                                        } else {
                                            128
                                        }
                                    })
                                    .collect();
                                let mask = Mask::from_parts(data, width, height);
                                if matches!(scene, Scene::LayerMask) {
                                    ctx.push_mask_layer(mask);
                                } else {
                                    ctx.set_mask(mask);
                                }
                                ctx.fill_rect(&full);
                                if matches!(scene, Scene::LayerMask) {
                                    ctx.pop_layer();
                                }
                            }
                            Scene::Gradient => {
                                // Two equal stops still use the indexed gradient pipeline,
                                // but give an analytic exact oracle at every sample.
                                ctx.set_paint(Gradient {
                                    kind: LinearGradientPosition {
                                        start: full.origin(),
                                        end: (full.x1, full.y1).into(),
                                    }
                                    .into(),
                                    stops: ColorStops::from(
                                        [ColorStop::from((0.0, RED)), ColorStop::from((1.0, RED))]
                                            .as_slice(),
                                    ),
                                    ..Default::default()
                                });
                                ctx.fill_rect(&full);
                            }
                            Scene::Image => {
                                let mut source = Pixmap::new(1, 1);
                                source.data_mut().fill(RED.premultiply().to_rgba8());
                                ctx.set_paint(Image {
                                    image: ImageSource::Pixmap(Arc::new(source)),
                                    sampler: ImageSampler::default(),
                                });
                                ctx.fill_rect(&full);
                            }
                            Scene::Empty | Scene::SrcOver => {}
                        }
                        ctx.flush();
                        ctx.render_with(
                            &mut image,
                            &mut resources,
                            RasterizerSettings {
                                render_mode,
                                target_init: if matches!(scene, Scene::SrcOver | Scene::SrcOverTail)
                                {
                                    TargetInit::SrcOver
                                } else {
                                    TargetInit::Clear(vello_cpu::color::AlphaColor::TRANSPARENT)
                                },
                                ..Default::default()
                            },
                        );
                        let label = format!(
                            "{width}x{height} {scene:?} {render_mode:?} threads={num_threads} {level:?}"
                        );
                        check_pixels(&image, &label, |x, y| {
                            let in_tail =
                                (if transpose { y } else { x }) >= usize::from(length - 3);
                            match scene {
                                Scene::Empty => CLEAR,
                                Scene::SrcOver => BLUE_PIXEL,
                                Scene::SrcOverTail if !in_tail => BLUE_PIXEL,
                                Scene::Opacity => [128, 0, 0, 128],
                                Scene::PaintMask | Scene::LayerMask if !in_tail => [128, 0, 0, 128],
                                Scene::TranslatedTail | Scene::NestedClip if !in_tail => CLEAR,
                                Scene::DepthOverlap if !in_tail => BLUE_PIXEL,
                                _ => RED_PIXEL,
                            }
                        });
                    }

                    // Resize down and back: retained tile/layer buffers must not retain pixels.
                    ctx.reset_with(vello_cpu::ResetOptions {
                        size: Some((3, 2)),
                        ..Default::default()
                    });
                    ctx.set_paint(RED);
                    ctx.fill_rect(&Rect::new(0.0, 0.0, 3.0, 2.0));
                    ctx.flush();
                    let mut small = Pixmap::new(3, 2);
                    let settings = RasterizerSettings {
                        render_mode,
                        ..Default::default()
                    };
                    ctx.render_with(&mut small, &mut resources, settings);
                    check_pixels(&small, "resized down", |_, _| RED_PIXEL);
                    ctx.reset_with(vello_cpu::ResetOptions {
                        size: Some((width, height)),
                        ..Default::default()
                    });
                    ctx.flush();
                    ctx.render_with(&mut image, &mut resources, settings);
                    check_pixels(&image, "resized back and empty", |_, _| CLEAR);
                }
            }
        }
    }
}

macro_rules! boundary_tests {
    ($($name:ident: $length:literal),+ $(,)?) => {$ (
        #[test]
        fn $name() {
            boundary($length);
        }
    )+};
}

boundary_tests! {
    boundary_65520: 65_520,
    boundary_65528: 65_528,
    boundary_65531: 65_531,
    boundary_65532: 65_532,
    boundary_65533: 65_533,
    boundary_65534: 65_534,
    boundary_65535: 65_535,
}

#[test]
#[cfg(any(feature = "std", target_arch = "wasm32"))]
fn native_simd_masks_and_indexed_paints_reach_last_frame_pixel() {
    let level = Level::new();
    eprintln!("native SIMD level: {level:?}");
    boundary_with(
        u16::MAX,
        level,
        &[1],
        &[
            Scene::PaintMask,
            Scene::LayerMask,
            Scene::Gradient,
            Scene::Image,
        ],
    );
}

#[test]
fn clip_touching_right_edge_leaves_target_clear() {
    for width in [100, 256, 300] {
        let mut ctx = context(width, 100, 0, Level::baseline());
        let mut image = Pixmap::new(width, 100);
        for y in [0.0, 2.0, 3.9, 4.0, 10.0, 49.0, 96.0] {
            let x = f64::from(width);
            let path = Rect::new(x, y, x + 0.5, y + 0.5).to_path(0.1);
            ctx.push_clip_path(&path);
            ctx.set_paint(RED);
            ctx.fill_path(&path);
            ctx.pop_clip();
        }
        ctx.flush();
        ctx.render_with(
            &mut image,
            &mut Resources::new(),
            RasterizerSettings::default(),
        );
        check_pixels(&image, "right edge", |_, _| CLEAR);
    }
}
