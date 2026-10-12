// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The clip of a filter layer applies where the filtered layer is composited.

use vello_cpu::color::palette::css::RED;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{BezPath, Rect, Shape};
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

fn render(filter: &Filter, combined: bool, mode: RenderMode) -> Pixmap {
    // Filters currently use the single-threaded dispatcher.
    let mut ctx = RenderContext::new_with(
        64,
        16,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );
    let mut active = BezPath::new();
    active.move_to((0.0, 0.0));
    active.line_to((56.0, 0.0));
    active.line_to((0.0, 16.0));
    active.close_path();
    ctx.push_clip_path(&active);
    let layer = Rect::new(8.0, 0.0, 40.0, 4.0).to_path(0.1);
    if combined {
        ctx.push_layer(Some(&layer), None, None, None, Some(filter.clone()));
    } else {
        ctx.push_clip_layer(&layer);
        ctx.push_filter_layer(filter.clone());
    }
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(4.0, 0.0, 60.0, 4.0));
    ctx.pop_layer();
    if !combined {
        ctx.pop_layer();
    }
    ctx.pop_clip();
    ctx.flush();
    let mut pixmap = Pixmap::new(64, 16);
    ctx.render_with(
        &mut pixmap,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: mode,
            ..Default::default()
        },
    );
    pixmap
}

#[test]
fn filter_layer_clip_matches_an_enclosing_clip_layer() {
    // A blur shifts its source viewport away from the parent's.
    let blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });
    for mode in modes() {
        let combined = render(&blur, true, mode);
        let separate = render(&blur, false, mode);
        assert!(
            combined.data().iter().any(|p| p.a != 0),
            "{mode:?}: empty result"
        );
        assert_eq!(
            combined.data_as_u8_slice(),
            separate.data_as_u8_slice(),
            "{mode:?}"
        );
    }
}
