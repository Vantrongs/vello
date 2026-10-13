// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A filter layer's pixmap covers the content its commands draw, wherever the
//! recorder draws it.

use vello_cpu::color::AlphaColor;
use vello_cpu::color::palette::css::BLUE;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{BezPath, Rect, Shape, Stroke};
use vello_cpu::peniko::Fill;
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
};

fn modes() -> impl Iterator<Item = RenderMode> {
    [
        #[cfg(feature = "u8_pipeline")]
        RenderMode::OptimizeSpeed,
        #[cfg(feature = "f32_pipeline")]
        RenderMode::OptimizeQuality,
    ]
    .into_iter()
}

fn flood() -> Filter {
    Filter::from_primitive(FilterPrimitive::Flood {
        color: AlphaColor::from_rgba8(200, 30, 60, 255),
    })
}

/// Render `draw` in a filter layer with `filter` in a `size` by `size` view.
fn render(
    size: u16,
    filter: &Filter,
    mode: RenderMode,
    draw: impl FnOnce(&mut RenderContext),
) -> Pixmap {
    render_nested(size, core::slice::from_ref(filter), mode, draw)
}

/// Render `draw` in filter layers with `filters`, each nested in the one
/// before, in a `size` by `size` view.
fn render_nested(
    size: u16,
    filters: &[Filter],
    mode: RenderMode,
    draw: impl FnOnce(&mut RenderContext),
) -> Pixmap {
    let mut context = RenderContext::new_with(
        size,
        size,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    for filter in filters {
        context.push_filter_layer(filter.clone());
    }
    context.set_paint(BLUE);
    draw(&mut context);
    for _ in filters {
        context.pop_layer();
    }
    context.flush();
    let mut pixmap = Pixmap::new(size, size);
    context.render_with(
        &mut pixmap,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: mode,
            ..Default::default()
        },
    );
    pixmap
}

fn line(from: (f64, f64), to: (f64, f64)) -> BezPath {
    let mut path = BezPath::new();
    path.move_to(from);
    path.line_to(to);
    path
}

#[test]
fn a_dash_between_gaps_at_both_ends_is_flooded() {
    let dashed = Stroke::new(1.0).with_dashes(500_000.0, [10.0, 999_990.0]);
    for mode in modes() {
        let actual = render(256, &flood(), mode, |context| {
            context.set_stroke(dashed.clone());
            context.stroke_path(&line((-500_000.0, 100.0), (500_000.0, 100.0)));
        });
        // The only dash the pattern draws.
        let expected = render(256, &flood(), mode, |context| {
            context.set_stroke(Stroke::new(1.0));
            context.stroke_path(&line((0.0, 100.0), (10.0, 100.0)));
        });
        assert!(expected.data().iter().any(|pixel| pixel.a != 0));
        assert!(actual.data() == expected.data(), "{mode:?}");
    }
}

#[test]
fn a_dash_drawn_on_from_a_closed_subpath_starts_the_pattern_over() {
    let filters = [
        Filter::from_primitive(FilterPrimitive::Offset {
            dx: -300_000.0,
            dy: 0.0,
        }),
        flood(),
    ];
    let dashed = Stroke::new(1.0).with_dashes(500_000.0, [10.0, 999_990.0]);
    let mut path = line((-200_000.0, 100.0), (50_000.0, 100.0));
    path.close_path();
    path.line_to((800_000.0, 100.0));
    for mode in modes() {
        let actual = render_nested(256, &filters, mode, |context| {
            context.set_stroke(dashed.clone());
            context.stroke_path(&path);
        });
        // Its last line starts where the closed subpath does, at the start of
        // the pattern, whose only dash it draws.
        let expected = render_nested(256, &filters, mode, |context| {
            context.set_stroke(Stroke::new(1.0));
            context.stroke_path(&line((300_000.0, 100.0), (300_010.0, 100.0)));
        });
        assert!(expected.data().iter().any(|pixel| pixel.a != 0));
        assert!(actual.data() == expected.data(), "{mode:?}");
    }
}

#[test]
fn a_dash_between_gaps_at_both_ends_is_blurred() {
    let blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 1.0,
        edge_mode: EdgeMode::None,
    });
    let dashed = Stroke::new(2.0).with_dashes(500_000.0, [100_000.0, 900_000.0]);
    for mode in modes() {
        let actual = render(64, &blur, mode, |context| {
            context.set_stroke(dashed.clone());
            context.stroke_path(&line((32.0, -500_000.0), (32.0, 500_000.0)));
        });
        let expected = render(64, &blur, mode, |context| {
            context.set_stroke(Stroke::new(2.0));
            context.stroke_path(&line((32.0, 0.0), (32.0, 100_000.0)));
        });
        assert!(expected.data().iter().any(|pixel| pixel.a != 0));
        assert!(actual.data() == expected.data(), "{mode:?}");
    }
}

#[test]
fn clip_paths_of_lines_traced_back_hide_everything() {
    let mut clip = BezPath::new();
    for x in [100.0, 200.0] {
        clip.move_to((x, 100.0));
        clip.line_to((x, 200.0));
        clip.close_path();
    }
    for fill_rule in [Fill::NonZero, Fill::EvenOdd] {
        for mode in modes() {
            let actual = render(256, &flood(), mode, |context| {
                context.set_fill_rule(fill_rule);
                context.push_clip_path(&clip);
                context.fill_rect(&Rect::new(140.0, 140.0, 160.0, 160.0));
                context.pop_clip();
            });
            assert!(
                actual.data().iter().all(|pixel| pixel.a == 0),
                "{fill_rule:?}, {mode:?}"
            );
        }
    }
}

#[test]
fn content_at_the_view_origin_rounds_as_the_recorder_rounds_it() {
    // The left side lies just left of the tile edge at 4 in `f32`.
    let left = 4.0 - 2_f64.powi(-22);
    for mode in modes() {
        let actual = render(256, &flood(), mode, |context| {
            context.fill_path(&Rect::new(left, 10.0, 8.0, 20.0).to_path(0.1));
        });
        let expected = render(256, &flood(), mode, |context| {
            context.fill_path(&Rect::new(1.0, 10.0, 8.0, 20.0).to_path(0.1));
        });
        assert!(actual.data() == expected.data(), "{mode:?}");
    }
}
