// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filters nested in each other, in clips and in blend layers show the same
//! pixels as the same part of a larger target.

use vello_cpu::color::AlphaColor;
use vello_cpu::color::palette::css::{BLUE, GOLD, RED};
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Circle, Rect, Shape, Vec2};
use vello_cpu::peniko::{BlendMode, Compose, Gradient, Mix};
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
};

const VIEW: u16 = 64;
/// How far the larger target reaches past the view on each side.
const MARGIN: u16 = 64;

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
enum Scene {
    NestedFilters,
    FilterInClip,
    ClipInFilter,
    BlendInFilter,
    FilterInBlend,
    /// A blur wide enough to read its source at a coarser resolution.
    WideBlur,
}

const SCENES: [Scene; 6] = [
    Scene::NestedFilters,
    Scene::FilterInClip,
    Scene::ClipInFilter,
    Scene::BlendInFilter,
    Scene::FilterInBlend,
    Scene::WideBlur,
];

fn blur(std_deviation: f32) -> Filter {
    Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    })
}

fn shadow(dx: f32, dy: f32, std_deviation: f32) -> Filter {
    Filter::from_primitive(FilterPrimitive::DropShadow {
        dx,
        dy,
        std_deviation,
        color: AlphaColor::from_rgba8(200, 30, 60, 200),
        edge_mode: EdgeMode::None,
    })
}

fn multiply() -> BlendMode {
    BlendMode::new(Mix::Multiply, Compose::SrcOver)
}

/// Draw `scene`, whose content and output straddle the edges of the view.
fn draw(ctx: &mut RenderContext, scene: Scene) {
    let square = |x: f64, y: f64, size: f64| Rect::new(x, y, x + size, y + size);
    match scene {
        Scene::NestedFilters => {
            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
                dx: 40.0,
                dy: -24.0,
            }));
            ctx.set_paint(Gradient::new_linear((-40.0, 0.0), (40.0, 0.0)).with_stops([BLUE, GOLD]));
            ctx.fill_rect(&square(-30.0, 40.0, 20.0));
            ctx.push_filter_layer(blur(3.0));
            ctx.set_paint(RED);
            ctx.fill_rect(&square(10.0, 60.0, 24.0));
            ctx.push_filter_layer(shadow(-6.0, 6.0, 2.0));
            ctx.set_paint(BLUE);
            ctx.fill_rect(&square(50.0, 30.0, 16.0));
            ctx.pop_layer();
            ctx.pop_layer();
            ctx.pop_layer();
        }
        Scene::FilterInClip => {
            ctx.push_clip_layer(&Circle::new((32.0, 32.0), 40.0).to_path(0.1));
            ctx.push_filter_layer(shadow(20.0, 20.0, 4.0));
            ctx.set_paint(BLUE);
            ctx.fill_rect(&square(-8.0, -8.0, 24.0));
            ctx.fill_rect(&square(50.0, 20.0, 20.0));
            ctx.pop_layer();
            ctx.pop_layer();
        }
        Scene::ClipInFilter => {
            ctx.push_filter_layer(blur(4.0));
            ctx.push_clip_layer(&Circle::new((60.0, 10.0), 20.0).to_path(0.1));
            ctx.set_paint(RED);
            ctx.fill_rect(&Rect::new(-10.0, -10.0, 80.0, 30.0));
            ctx.pop_layer();
            ctx.set_paint(BLUE);
            ctx.fill_rect(&square(-6.0, 44.0, 16.0));
            ctx.pop_layer();
        }
        Scene::BlendInFilter => {
            ctx.push_filter_layer(shadow(-12.0, 10.0, 3.0));
            ctx.set_paint(GOLD);
            ctx.fill_rect(&square(-10.0, 20.0, 30.0));
            ctx.push_blend_layer(multiply());
            ctx.set_paint(BLUE);
            ctx.fill_rect(&square(4.0, 30.0, 30.0));
            ctx.pop_layer();
            ctx.pop_layer();
        }
        Scene::FilterInBlend => {
            ctx.set_paint(GOLD);
            ctx.fill_rect(&Rect::new(-20.0, -20.0, 84.0, 40.0));
            ctx.push_blend_layer(multiply());
            ctx.push_filter_layer(blur(5.0));
            ctx.set_paint(BLUE);
            ctx.fill_rect(&square(40.0, 20.0, 32.0));
            ctx.pop_layer();
            ctx.pop_layer();
        }
        Scene::WideBlur => {
            ctx.push_filter_layer(blur(100.0));
            ctx.set_paint(BLUE);
            ctx.fill_rect(&square(20.0, 20.0, 24.0));
            ctx.fill_rect(&square(56.0, -8.0, 16.0));
            ctx.pop_layer();
        }
    }
}

/// Render the part of `scene` that starts at `origin` into a target of `size`.
fn render(size: u16, origin: Vec2, scene: Scene, mode: RenderMode) -> Pixmap {
    // Filters currently use the single-threaded dispatcher.
    let mut ctx = RenderContext::new_with(
        size,
        size,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    ctx.set_transform(Affine::translate(-origin));
    draw(&mut ctx, scene);
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

fn crop(image: &Pixmap, x: usize, y: usize) -> Vec<u8> {
    let row = usize::from(image.width()) * 4;
    let view = usize::from(VIEW) * 4;
    image
        .data_as_u8_slice()
        .chunks_exact(row)
        .skip(y)
        .take(usize::from(VIEW))
        .flat_map(|line| &line[x * 4..x * 4 + view])
        .copied()
        .collect()
}

#[test]
fn composed_filters_match_a_larger_target() {
    let margin = f64::from(MARGIN);
    for mode in modes() {
        for scene in SCENES {
            let large = render(VIEW + 2 * MARGIN, Vec2::new(-margin, -margin), scene, mode);
            let actual = render(VIEW, Vec2::ZERO, scene, mode);
            assert!(actual.data().iter().any(|pixel| pixel.a != 0), "{scene:?}");
            let expected = crop(&large, usize::from(MARGIN), usize::from(MARGIN));
            assert!(
                actual.data_as_u8_slice() == expected.as_slice(),
                "{scene:?}, {mode:?}"
            );
        }
    }
}

fn context() -> RenderContext {
    RenderContext::new_with(
        VIEW,
        VIEW,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    )
}

fn finish(mut ctx: RenderContext, mode: RenderMode) -> Vec<u8> {
    ctx.flush();
    let mut image = Pixmap::new(VIEW, VIEW);
    ctx.render_with(
        &mut image,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: mode,
            ..Default::default()
        },
    );
    image.data_as_u8_slice().to_vec()
}

#[test]
fn clip_pushed_in_a_filter_layer_applies_after_it() {
    let circle = Circle::new((32.0, 32.0), 20.0).to_path(0.1);
    let square = Rect::new(8.0, 8.0, 56.0, 56.0);
    for mode in modes() {
        let mut ctx = context();
        ctx.push_filter_layer(blur(2.0));
        ctx.push_clip_path(&circle);
        ctx.pop_layer();
        ctx.set_paint(BLUE);
        ctx.fill_rect(&square);
        ctx.pop_clip();
        let actual = finish(ctx, mode);

        let mut ctx = context();
        ctx.push_clip_path(&circle);
        ctx.set_paint(BLUE);
        ctx.fill_rect(&square);
        ctx.pop_clip();
        let expected = finish(ctx, mode);
        assert_eq!(&expected[(10 * 64 + 10) * 4..][..4], [0; 4]);
        assert!(actual == expected, "{mode:?}");
    }
}

#[test]
fn reset_drops_an_open_filter_layer() {
    let square = Rect::new(8.0, 8.0, 24.0, 24.0);
    for mode in modes() {
        let mut ctx = context();
        ctx.push_filter_layer(blur(2.0));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 64.0, 64.0));
        ctx.reset();
        ctx.set_paint(BLUE);
        ctx.fill_rect(&square);
        let actual = finish(ctx, mode);

        let mut ctx = context();
        ctx.set_paint(BLUE);
        ctx.fill_rect(&square);
        assert!(actual == finish(ctx, mode), "{mode:?}");
    }
}
