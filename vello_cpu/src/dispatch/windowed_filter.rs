// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter layers whose source does not fit the `u16` coordinate domain.
//!
//! Strips, tiles and fine rasterization address pixels with `u16`. A filter can
//! still need source pixels beyond that domain: its halo reaches past a target
//! that is 65535 pixels wide, or an offset moves geometry from far away into
//! view. Such a layer records its commands instead of drawing them. When it is
//! popped, they are replayed into windows of at most 65532 pixels that cover the
//! source regions the visible output depends on: one around the view, or for
//! an offset or drop shadow, the view and the part its moved copy reads. At
//! rasterization the windows are stitched into one `u32`-sized pixmap that is
//! filtered as a whole, or assembled like any offset or drop shadow layer (see
//! [`DisplacedPlan`]), and the part inside the parent viewport is composited
//! like any other filter layer.
//!
//! Filters nested inside are rendered the same way and handed to the windows as
//! finished pixmaps, so every filter sees its whole source at once and the
//! result does not depend on where the window edges fall.

use super::Dispatcher;
use super::single_threaded::SingleThreadedDispatcher;
use crate::filter::context::ScratchBuffer;
use crate::filter::displaced::{self, DisplacedPlan};
use crate::filter::placed::{Bounds, PlacedPixmap};
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Rect, Stroke, Vec2};
use crate::peniko::{BlendMode, Fill};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::clip::PushedClip;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::tile::Tile;

/// Width and height of a window, the largest tile multiple in the `u16` domain.
const WINDOW_SIZE: i64 = 65532;

/// Return the regions of a filter's source space that its output inside
/// `view` depends on. `view` is in the space of the filter's parent. Regions
/// that overlap are merged, so every source pixel is rendered once.
///
/// # Panics
///
/// Panics if a region, or the filter pixmap that can be allocated in it,
/// leaves the `u32` domain.
fn source_regions(filter: &FilterData, view: Bounds) -> Vec<Bounds> {
    let shift = filter.source_shift();
    let view = view.translate((i64::from(shift.0), i64::from(shift.1)));
    let (regions, pixmap) = match filter.displacement() {
        Some(displacement) => {
            let [kept, moved] = displaced::source_regions(displacement, view);
            let regions = if kept.intersect(moved).is_empty() {
                vec![kept, moved]
            } else {
                vec![kept.union(moved)]
            };
            (regions, moved.expand(displacement.blur_padding))
        }
        None => {
            let region = view.expand(filter.source_padding);
            (vec![region], region.expand(filter.filter_padding))
        }
    };
    let regions: Vec<Bounds> = regions.into_iter().filter(|r| !r.is_empty()).collect();
    let end = pixmap.snap_to_tiles();
    assert!(
        regions
            .iter()
            .all(|region| region.x0 >= 0 && region.y0 >= 0)
            && end.x1 <= i64::from(u32::MAX)
            && end.y1 <= i64::from(u32::MAX),
        "filter source viewport exceeds u32 coordinate domain"
    );
    regions
}

/// A clip as it was pushed, in root target coordinates.
#[derive(Clone, Debug)]
pub(crate) enum Clip {
    Path {
        path: BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    },
    Rect(Rect),
}

impl Clip {
    pub(crate) fn from_pushed(clip: PushedClip<'_>) -> Self {
        match clip {
            PushedClip::Path {
                elements,
                fill_rule,
                transform,
                aliasing_threshold,
            } => Self::Path {
                path: BezPath::from_vec(elements.to_vec()),
                fill_rule,
                transform,
                aliasing_threshold,
            },
            PushedClip::Rect(rect) => Self::Rect(rect),
        }
    }

    /// Push the clip into `target`, whose pixels are root target pixels moved by `shift`.
    fn push_into(&self, target: &mut SingleThreadedDispatcher, shift: Vec2) {
        match self {
            Self::Path {
                path,
                fill_rule,
                transform,
                aliasing_threshold,
            } => target.push_clip_path(
                path,
                *fill_rule,
                Affine::translate(shift) * *transform,
                *aliasing_threshold,
            ),
            Self::Rect(rect) => target.push_clip_rect(&(*rect + shift)),
        }
    }
}

/// Arguments of a layer pushed inside a windowed filter layer.
#[derive(Debug)]
pub(crate) struct Layer {
    pub(crate) clip_path: Option<BezPath>,
    pub(crate) fill_rule: Fill,
    pub(crate) clip_transform: Affine,
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) aliasing_threshold: Option<u8>,
    pub(crate) mask: Option<Mask>,
    pub(crate) filter: Option<FilterData>,
}

impl Layer {
    /// Push the layer, which has no filter, into `target`.
    fn push_into(&self, target: &mut SingleThreadedDispatcher, to_window: Affine) {
        target.push_layer(
            self.clip_path.as_ref(),
            self.fill_rule,
            to_window * self.clip_transform,
            self.blend_mode,
            self.opacity,
            self.aliasing_threshold,
            self.mask.clone(),
            None,
        );
    }
}

/// A dispatcher call made inside a windowed filter layer.
#[derive(Debug)]
pub(crate) enum Command {
    FillPath {
        path: BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    },
    StrokePath {
        path: BezPath,
        stroke: Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    },
    FillRect {
        rect: Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    },
    PushLayer(Layer),
    PopLayer,
    PushClip(Clip),
    PopClip,
}

/// A windowed filter layer whose commands are being recorded.
#[derive(Debug)]
pub(crate) struct Recording {
    filter: FilterData,
    /// Where the output is visible, in the space of the filter's parent.
    view: Bounds,
    /// Translation from root target coordinates to the filter's source space.
    root_shift: (i64, i64),
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    commands: Vec<Command>,
    /// Source regions of the layers open inside the recording, innermost last;
    /// `None` for layers without a filter.
    open_layers: Vec<Option<Vec<Bounds>>>,
    /// Source regions of the recorded layer itself.
    regions: Vec<Bounds>,
}

impl Recording {
    /// Start recording a filter layer whose output is visible in `view`.
    ///
    /// # Panics
    ///
    /// Panics before recording anything if a source region leaves the `u32` domain.
    pub(crate) fn new(
        filter: FilterData,
        view: Bounds,
        parent_root_shift: (i64, i64),
        clips: Vec<Clip>,
    ) -> Self {
        let regions = source_regions(&filter, view);
        let shift = filter.source_shift();
        Self {
            filter,
            view,
            root_shift: (
                parent_root_shift.0 + i64::from(shift.0),
                parent_root_shift.1 + i64::from(shift.1),
            ),
            clips,
            commands: Vec::new(),
            open_layers: Vec::new(),
            regions,
        }
    }

    pub(crate) fn record(&mut self, command: Command) {
        self.commands.push(command);
    }

    /// Record a layer push. A nested filter's regions are checked before anything is recorded.
    pub(crate) fn push_layer(&mut self, layer: Layer) {
        let regions = layer.filter.as_ref().map(|filter| {
            let enclosing = self
                .open_layers
                .iter()
                .rev()
                .find_map(Option::as_ref)
                .unwrap_or(&self.regions);
            enclosing
                .iter()
                .flat_map(|view| source_regions(filter, *view))
                .collect()
        });
        self.open_layers.push(regions);
        self.record(Command::PushLayer(layer));
    }

    /// Record a layer pop, or return `false` if it ends the recorded layer itself.
    pub(crate) fn pop_layer(&mut self) -> bool {
        if self.open_layers.pop().is_none() {
            return false;
        }
        self.record(Command::PopLayer);
        true
    }

    /// Replay the recorded commands into windows.
    pub(crate) fn finish(self, level: Level) -> WindowedFilter {
        WindowedFilter::new(
            &self.commands,
            self.filter,
            self.view,
            self.root_shift,
            &self.clips,
            level,
        )
    }
}

/// A recorded command, or a nested filter layer rendered on its own.
enum Step<'a> {
    Command(&'a Command),
    Nested { layer: &'a Layer, index: usize },
}

/// A window of a filter's source space, replayed into its own dispatcher.
#[derive(Debug)]
struct Window {
    origin: (i64, i64),
    dispatcher: SingleThreadedDispatcher,
}

impl Window {
    /// Render the window content, or return `None` if it is empty. `nested`
    /// holds the outputs of the filters nested in this window's replay.
    fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        nested: &[PlacedPixmap],
    ) -> Option<PlacedPixmap> {
        let encoded_paints = paints_at(encoded_paints, self.origin);
        let (pixmap, rect) =
            self.dispatcher
                .render_window::<S, F>(simd, &encoded_paints, image_resolver, nested)?;
        Some(PlacedPixmap {
            pixmap,
            bounds: Bounds::of_rect(rect, self.origin),
        })
    }
}

/// Return copies of `paints` for sampling at pixels relative to `origin`.
fn paints_at(paints: &[EncodedPaint], (x, y): (i64, i64)) -> Vec<EncodedPaint> {
    let to_source = Affine::translate((x as f64, y as f64));
    paints
        .iter()
        .map(|paint| {
            let mut paint = paint.clone();
            let transform = match &mut paint {
                EncodedPaint::Gradient(gradient) => &mut gradient.transform,
                EncodedPaint::Image(image) => &mut image.transform,
                EncodedPaint::BlurredRoundedRect(rect) => &mut rect.transform,
            };
            *transform *= to_source;
            paint
        })
        .collect()
}

/// The windows of one source region and the filters nested in them.
#[derive(Debug)]
struct Replay {
    nested: Vec<WindowedFilter>,
    windows: Vec<Window>,
}

/// How the windows of a filter become its output.
#[derive(Debug)]
enum Output {
    /// Stitched into one pixmap with these bounds, which is filtered as a whole.
    Whole(Bounds),
    /// Assembled from the parts the visible output reads.
    Displaced(DisplacedPlan),
}

/// A filter layer rendered in windows, see the module documentation.
#[derive(Debug)]
pub(crate) struct WindowedFilter {
    filter: FilterData,
    replays: Vec<Replay>,
    output: Output,
    /// Translation from the filter's source space to its parent's space.
    to_parent: (i64, i64),
}

impl WindowedFilter {
    fn new(
        commands: &[Command],
        filter: FilterData,
        view: Bounds,
        root_shift: (i64, i64),
        clips: &[Clip],
        level: Level,
    ) -> Self {
        let replays: Vec<Replay> = source_regions(&filter, view)
            .into_iter()
            .map(|region| Replay::new(commands, region, root_shift, clips, level))
            .collect();
        let content = replays
            .iter()
            .flat_map(|replay| &replay.windows)
            .fold(Bounds::EMPTY, |content, window| {
                content.union(window.dispatcher.window_content(window.origin))
            });

        let shift = filter.source_shift();
        let shift = (i64::from(shift.0), i64::from(shift.1));
        let output = match filter.displacement() {
            Some(displacement) => {
                let view = view.translate(shift);
                Output::Displaced(DisplacedPlan::new(displacement, content, view))
            }
            None => {
                // Like a filter layer in a `u16` viewport, keep the pixmap inside
                // the non-negative source space so both decimate identically.
                let pixmap = content.expand(filter.filter_padding);
                let pixmap = Bounds::new(pixmap.x0.max(0), pixmap.y0.max(0), pixmap.x1, pixmap.y1);
                Output::Whole(pixmap.snap_to_tiles())
            }
        };

        Self {
            filter,
            replays,
            output,
            to_parent: (-shift.0, -shift.1),
        }
    }

    /// Bounds of the filtered pixmap in the parent's space.
    pub(crate) fn output_bounds(&self) -> Bounds {
        let bounds = match &self.output {
            Output::Whole(bounds) => *bounds,
            Output::Displaced(plan) => plan.output(),
        };
        bounds.translate(self.to_parent)
    }

    /// Return where the visible part of the output is composited in `view`, which
    /// starts at the origin of the parent's space.
    pub(crate) fn placement_in(&self, view: Bounds) -> RectU16 {
        self.output_bounds()
            .intersect(view)
            .snap_to_tiles()
            .to_rect_u16((0, 0))
    }

    /// Render and filter the layer. The output is in the parent's space.
    pub(crate) fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> PlacedPixmap {
        let nested: Vec<Vec<PlacedPixmap>> = self
            .replays
            .iter()
            .map(|replay| {
                replay
                    .nested
                    .iter()
                    .map(|filter| filter.render::<S, F>(simd, encoded_paints, image_resolver))
                    .collect()
            })
            .collect();
        let pieces = self
            .replays
            .iter()
            .zip(&nested)
            .flat_map(|(replay, nested)| {
                replay.windows.iter().filter_map(|window| {
                    window.render::<S, F>(simd, encoded_paints, image_resolver, nested)
                })
            });
        let filter = PreparedFilter::new(&self.filter.filter, &self.filter.transform);
        let mut scratch = ScratchBuffer::new();
        let output = match &self.output {
            Output::Whole(bounds) => {
                let mut pixmap = PlacedPixmap::new(*bounds);
                for piece in pieces {
                    pixmap.copy_from(&piece, piece.bounds);
                }
                if !bounds.is_empty() {
                    F::filter_layer(&mut pixmap.pixmap, filter, &mut scratch);
                }
                pixmap
            }
            Output::Displaced(plan) => plan.assemble(pieces, |pixmap| {
                F::filter_layer(pixmap, filter, &mut scratch);
            }),
        };
        output.translate(self.to_parent)
    }
}

impl Replay {
    /// Replay `commands` into the windows covering `region`.
    fn new(
        commands: &[Command],
        region: Bounds,
        root_shift: (i64, i64),
        clips: &[Clip],
        level: Level,
    ) -> Self {
        let mut nested = Vec::new();
        let steps = plan(commands, region, root_shift, clips, level, &mut nested);
        let windows = window_grid(region)
            .map(|window| replay(window, &steps, &nested, root_shift, clips, level))
            .collect();
        Self { nested, windows }
    }
}

/// Replace nested filter layers in `commands` by steps that composite their
/// output inside `region`, which is rendered separately and appended to `nested`.
fn plan<'a>(
    commands: &'a [Command],
    region: Bounds,
    root_shift: (i64, i64),
    clips: &[Clip],
    level: Level,
    nested: &mut Vec<WindowedFilter>,
) -> Vec<Step<'a>> {
    let mut steps = Vec::new();
    let mut active_clips = clips.to_vec();
    let mut index = 0;
    while index < commands.len() {
        let command = &commands[index];
        if let Command::PushLayer(layer) = command
            && let Some(filter) = &layer.filter
        {
            let end = matching_pop(commands, index);
            let shift = filter.source_shift();
            let filter_root_shift = (
                root_shift.0 + i64::from(shift.0),
                root_shift.1 + i64::from(shift.1),
            );
            nested.push(WindowedFilter::new(
                &commands[index + 1..end],
                filter.clone(),
                region,
                filter_root_shift,
                &active_clips,
                level,
            ));
            steps.push(Step::Nested {
                layer,
                index: nested.len() - 1,
            });
            // Clips pushed or popped inside the filter layer stay in effect after it.
            for command in &commands[index + 1..end] {
                track_clips(command, &mut active_clips);
                if matches!(command, Command::PushClip(_) | Command::PopClip) {
                    steps.push(Step::Command(command));
                }
            }
            index = end + 1;
            continue;
        }
        track_clips(command, &mut active_clips);
        steps.push(Step::Command(command));
        index += 1;
    }
    steps
}

fn track_clips(command: &Command, clips: &mut Vec<Clip>) {
    match command {
        Command::PushClip(clip) => clips.push(clip.clone()),
        Command::PopClip => {
            clips.pop();
        }
        _ => {}
    }
}

/// Return the index of the `PopLayer` that closes the layer pushed at `push`.
fn matching_pop(commands: &[Command], push: usize) -> usize {
    let mut depth = 0_usize;
    for (index, command) in commands.iter().enumerate().skip(push) {
        match command {
            Command::PushLayer(_) => depth += 1,
            Command::PopLayer => {
                depth -= 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => {}
        }
    }
    unreachable!("a finished recording closes every layer it opens")
}

/// Split `region` into windows on a tile-aligned grid.
fn window_grid(region: Bounds) -> impl Iterator<Item = Bounds> {
    let x0 = region.x0.div_euclid(i64::from(Tile::WIDTH)) * i64::from(Tile::WIDTH);
    let y0 = region.y0.div_euclid(i64::from(Tile::HEIGHT)) * i64::from(Tile::HEIGHT);
    let rows = (y0..region.y1).step_by(WINDOW_SIZE as usize);
    rows.flat_map(move |y| {
        (x0..region.x1).step_by(WINDOW_SIZE as usize).map(move |x| {
            Bounds::new(
                x,
                y,
                (x + WINDOW_SIZE).min(region.x1),
                (y + WINDOW_SIZE).min(region.y1),
            )
        })
    })
}

/// A filter without reach, which is never applied. Wrapping a window's content
/// in it records the tight bounds of that content and renders it into its own pixmap.
pub(crate) fn content_filter() -> FilterData {
    FilterData::new(
        Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 0.0,
            edge_mode: EdgeMode::None,
        }),
        Affine::IDENTITY,
    )
}

fn replay(
    window: Bounds,
    steps: &[Step<'_>],
    nested: &[WindowedFilter],
    root_shift: (i64, i64),
    clips: &[Clip],
    level: Level,
) -> Window {
    let origin = (window.x0, window.y0);
    let shift = Vec2::new(-origin.0 as f64, -origin.1 as f64);
    let clip_shift = Vec2::new(
        (root_shift.0 - origin.0) as f64,
        (root_shift.1 - origin.1) as f64,
    );
    let dispatcher = SingleThreadedDispatcher::window(
        u16::try_from(window.width()).unwrap(),
        u16::try_from(window.height()).unwrap(),
        level,
        (origin.0 - root_shift.0, origin.1 - root_shift.1),
        |target| {
            for clip in clips {
                clip.push_into(target, clip_shift);
            }
            for step in steps {
                match step {
                    Step::Command(command) => apply(target, command, shift, clip_shift),
                    Step::Nested { layer, index } => {
                        inject(target, layer, &nested[*index], *index, window, shift);
                    }
                }
            }
        },
    );
    Window { origin, dispatcher }
}

/// Composite the part of a nested filter's output that falls into `window`.
fn inject(
    target: &mut SingleThreadedDispatcher,
    layer: &Layer,
    filter: &WindowedFilter,
    index: usize,
    window: Bounds,
    shift: Vec2,
) {
    let origin = (window.x0, window.y0);
    let dest = filter
        .output_bounds()
        .intersect(window)
        .snap_to_tiles()
        .to_rect_u16(origin);
    target.push_rendered_filter_layer(
        layer,
        Affine::translate(shift),
        filter.filter.clone(),
        dest,
        index,
        origin,
    );
}

fn apply(target: &mut SingleThreadedDispatcher, command: &Command, shift: Vec2, clip_shift: Vec2) {
    let to_window = Affine::translate(shift);
    match command {
        Command::FillPath {
            path,
            fill_rule,
            transform,
            paint,
            blend_mode,
            aliasing_threshold,
            mask,
        } => target.fill_path(
            path,
            *fill_rule,
            to_window * *transform,
            paint.clone(),
            *blend_mode,
            *aliasing_threshold,
            mask.clone(),
        ),
        Command::StrokePath {
            path,
            stroke,
            transform,
            paint,
            blend_mode,
            aliasing_threshold,
            mask,
        } => target.stroke_path(
            path,
            stroke,
            to_window * *transform,
            paint.clone(),
            *blend_mode,
            *aliasing_threshold,
            mask.clone(),
        ),
        Command::FillRect {
            rect,
            paint,
            blend_mode,
            mask,
        } => target.fill_rect_fast(&(*rect + shift), paint.clone(), *blend_mode, mask.clone()),
        Command::PushLayer(layer) => layer.push_into(target, to_window),
        Command::PopLayer => target.pop_layer(),
        Command::PushClip(clip) => clip.push_into(target, clip_shift),
        Command::PopClip => target.pop_clip(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::color::palette::css::RED;

    #[test]
    fn source_regions_do_not_grow_with_the_offset() {
        let view = Bounds::of_viewport(12, 12);
        for distance in [65_536.0, 262_144.0, 16_777_216.0] {
            for (dx, dy) in [(distance, distance), (-distance, 4.0), (4.0, -distance)] {
                let shadow = FilterData::new(
                    Filter::from_primitive(FilterPrimitive::DropShadow {
                        dx,
                        dy,
                        std_deviation: 8.0,
                        color: RED,
                        edge_mode: EdgeMode::None,
                    }),
                    Affine::IDENTITY,
                );
                let regions = source_regions(&shadow, view);
                // The view, kept unmoved, and the view moved back with the blur's reach.
                let sizes: Vec<_> = regions.iter().map(|r| (r.width(), r.height())).collect();
                assert_eq!(sizes, [(12, 12), (60, 60)], "{dx}, {dy}");
            }
        }
    }
}
