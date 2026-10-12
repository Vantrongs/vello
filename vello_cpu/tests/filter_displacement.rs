// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Offsets and drop shadows produce the same visible pixels however far they move the source.

use vello_cpu::color::AlphaColor;
use vello_cpu::color::palette::css::BLUE;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Rect, Vec2};
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
};

const VIEW: u16 = 64;
const SOURCE_SIZE: f64 = 16.0;
/// A source that straddles the left edge of the view.
const NEAR_SOURCE: Vec2 = Vec2::new(-8.0, 24.0);
/// Where the moved copy of a far source lands, straddling the right edge.
const COPY_SPOT: Vec2 = Vec2::new(56.0, 4.0);
/// Offsets, all multiples of a tile, so that the source keeps its position
/// within tiles and the blur its decimation phase. The first moves both copies
/// out of view; the later ones exceed the `u16` domain.
const OFFSETS: [i64; 5] = [96, 4_096, 60_000, 262_144, 16_777_216];

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
    Offset,
    Shadow(f32),
    ShadowOnly(f32),
}

const EFFECTS: [Effect; 5] = [
    Effect::Offset,
    Effect::Shadow(0.0),
    Effect::Shadow(6.0),
    Effect::ShadowOnly(1.5),
    Effect::ShadowOnly(6.0),
];

fn filter(effect: Effect, (dx, dy): (i64, i64)) -> Filter {
    let (dx, dy) = (dx as f32, dy as f32);
    let color = AlphaColor::from_rgba8(200, 30, 60, 200);
    Filter::from_primitive(match effect {
        Effect::Offset => FilterPrimitive::Offset { dx, dy },
        Effect::Shadow(std_deviation) => FilterPrimitive::DropShadow {
            dx,
            dy,
            std_deviation,
            color,
            edge_mode: EdgeMode::None,
        },
        Effect::ShadowOnly(std_deviation) => FilterPrimitive::DropShadowOnly {
            dx,
            dy,
            std_deviation,
            color,
            edge_mode: EdgeMode::None,
        },
    })
}

/// Render the part of a scene that starts at `origin`: source rectangles at
/// `sources` in a layer with `effect` moved by `offset`.
fn render(
    size: u16,
    origin: Vec2,
    sources: &[Vec2],
    effect: Effect,
    offset: (i64, i64),
    mode: RenderMode,
) -> Pixmap {
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
    ctx.push_filter_layer(filter(effect, offset));
    ctx.set_paint(BLUE);
    for source in sources {
        ctx.fill_rect(&Rect::from_origin_size(
            source.to_point(),
            (SOURCE_SIZE, SOURCE_SIZE),
        ));
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

fn directions(distance: i64) -> [(i64, i64); 4] {
    [
        (distance, distance),
        (-distance, distance),
        (distance, -distance),
        (-distance, -distance),
    ]
}

/// Render a source partly in view and a far source whose copy lands partly in view.
fn near_and_far(effect: Effect, offset: (i64, i64), mode: RenderMode) -> Pixmap {
    let far = COPY_SPOT - Vec2::new(offset.0 as f64, offset.1 as f64);
    render(VIEW, Vec2::ZERO, &[NEAR_SOURCE, far], effect, offset, mode)
}

#[test]
fn visible_output_does_not_depend_on_the_distance() {
    for mode in modes() {
        for effect in EFFECTS {
            for direction in 0..4 {
                let reference = near_and_far(effect, directions(OFFSETS[0])[direction], mode);
                assert!(reference.data().iter().any(|pixel| pixel.a != 0));
                for distance in &OFFSETS[1..] {
                    let offset = directions(*distance)[direction];
                    let actual = near_and_far(effect, offset, mode);
                    assert!(
                        actual.data_as_u8_slice() == reference.data_as_u8_slice(),
                        "{effect:?}, offset={offset:?}, {mode:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn partly_visible_source_and_copy_match_a_larger_target() {
    // The source straddles one edge of the view, its copy the opposite one.
    for mode in modes() {
        for effect in EFFECTS {
            for (dx, dy) in directions(64) {
                let source = Vec2::new(
                    if dx > 0 { -8.0 } else { 56.0 },
                    if dy > 0 { -8.0 } else { 56.0 },
                );
                let margin = Vec2::new(64.0, 64.0);
                let large = render(192, -margin, &[source], effect, (dx, dy), mode);
                let actual = render(VIEW, Vec2::ZERO, &[source], effect, (dx, dy), mode);
                assert!(
                    actual.data_as_u8_slice() == crop(&large, 64, 64).as_slice(),
                    "{effect:?}, offset=({dx}, {dy}), {mode:?}"
                );
            }
        }
    }
}
