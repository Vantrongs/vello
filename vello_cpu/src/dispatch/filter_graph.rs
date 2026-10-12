// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter layers, rendered from the parts of their source that their visible
//! output reads.
//!
//! A filter layer records its commands instead of drawing them. When it is
//! popped, the regions of its source that its output inside the view reads
//! are planned backwards through every filter nested in it, see [`FilterPlan`].
//! A nested filter has to produce its output where its parent reads its
//! source, so its output regions are the regions all of its parent's outputs
//! read, merged. Each region is therefore planned and rendered once however
//! many outputs read it, and the work follows the view, not the extent of the
//! source, the distance it is moved or how deeply filters are nested.
//!
//! The commands are then replayed into windows that cover those regions, in
//! the pixel grid each region is read in: a wide blur reads a coarser source,
//! so its source and the filters nested in it are replayed at that
//! resolution. Strips, tiles and fine rasterization address pixels with `u16`,
//! so a window is at most 65532 pixels wide; regions and windows themselves
//! are placed in `i64` coordinates of the root target.
//!
//! Nested filters are rendered first and handed to the windows of their parent
//! as finished pixmaps, so every filter sees all of the source it reads and
//! the result does not depend on where window edges fall.

use super::Dispatcher;
use super::single_threaded::SingleThreadedDispatcher;
use crate::RasterizerSettings;
use crate::filter::context::ScratchBuffer;
use crate::filter::placed::{Bounds, PlacedPixmap, Regions};
use crate::filter::plan::FilterPlan;
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use vello_common::clip::PushedClip;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::FilterData;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;
use vello_common::tile::Tile;

/// Width and height of a window, the largest tile multiple in the `u16` domain.
const WINDOW_SIZE: i64 = 65532;

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

    /// Push the clip into `target`, whose pixels are root target pixels mapped by `to_window`.
    fn push_into(&self, target: &mut SingleThreadedDispatcher, to_window: Affine) {
        match self {
            Self::Path {
                path,
                fill_rule,
                transform,
                aliasing_threshold,
            } => target.push_clip_path(
                path,
                *fill_rule,
                to_window * *transform,
                *aliasing_threshold,
            ),
            Self::Rect(rect) => target.push_clip_rect(&to_window.transform_rect_bbox(*rect)),
        }
    }
}

/// Arguments of a layer pushed inside a filter layer.
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

/// A dispatcher call made inside a filter layer.
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

/// A filter layer whose commands are being recorded. While it is open, the
/// render context sends its commands here instead of to the dispatcher it is
/// drawn into, see [`Dispatcher::record_filter_layer`].
#[derive(Debug)]
pub(crate) struct Recording {
    filter: FilterData,
    /// Where the output is visible, in root target coordinates.
    view: Bounds,
    /// Translation from root target coordinates to the filter's source space.
    root_shift: (i64, i64),
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    commands: Vec<Command>,
    /// The number of open layers, the recorded layer included.
    depth: usize,
}

impl Recording {
    /// Start recording a filter layer whose output is visible in `view`.
    pub(crate) fn new(filter: FilterData, view: Bounds, clips: Vec<Clip>) -> Self {
        let shift = filter.source_shift();
        Self {
            filter,
            view,
            root_shift: (i64::from(shift.0), i64::from(shift.1)),
            clips,
            commands: Vec::new(),
            depth: 1,
        }
    }

    /// Push the clips recorded in the layer into `target`, in order.
    pub(crate) fn push_clips_into(&self, target: &mut SingleThreadedDispatcher) {
        for command in &self.commands {
            match command {
                Command::PushClip(clip) => clip.push_into(target, Affine::IDENTITY),
                Command::PopClip => target.pop_clip(),
                _ => {}
            }
        }
    }

    /// Plan the recorded layer and replay its commands into windows.
    pub(crate) fn finish(self, level: Level) -> FilterGraph {
        let mut layers = Vec::new();
        parse(
            &self.commands,
            0..self.commands.len(),
            self.filter,
            self.root_shift,
            self.clips,
            &mut layers,
        );
        FilterGraph::new(&self.commands, &layers, self.view, level)
    }
}

impl Dispatcher for Recording {
    fn has_layers(&self) -> bool {
        self.depth > 0
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        self.commands.push(Command::FillPath {
            path: path.clone(),
            fill_rule,
            transform,
            paint,
            blend_mode,
            aliasing_threshold,
            mask,
        });
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        self.commands.push(Command::StrokePath {
            path: path.clone(),
            stroke: stroke.clone(),
            transform,
            paint,
            blend_mode,
            aliasing_threshold,
            mask,
        });
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        self.commands.push(Command::FillRect {
            rect: *rect,
            paint,
            blend_mode,
            mask,
        });
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.commands.push(Command::PushClip(Clip::Path {
            path: path.clone(),
            fill_rule,
            transform,
            aliasing_threshold,
        }));
    }

    fn push_clip_rect(&mut self, rect: &Rect) {
        self.commands.push(Command::PushClip(Clip::Rect(*rect)));
    }

    fn pop_clip(&mut self) {
        self.commands.push(Command::PopClip);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter_data: Option<FilterData>,
    ) {
        self.depth += 1;
        self.commands.push(Command::PushLayer(Layer {
            clip_path: clip_path.cloned(),
            fill_rule,
            clip_transform,
            blend_mode,
            opacity,
            aliasing_threshold,
            mask,
            filter: filter_data,
        }));
    }

    fn pop_layer(&mut self) {
        self.depth -= 1;
        // The recorded layer itself is popped by the dispatcher it is drawn into.
        if self.depth > 0 {
            self.commands.push(Command::PopLayer);
        }
    }

    fn reset(&mut self, _: u16, _: u16) {
        unreachable!("the render context drops a recording when it is reset");
    }

    fn flush(&mut self) {}

    fn rasterize(
        &self,
        _: PixmapMut<'_>,
        _: u16,
        _: u16,
        _: RasterizerSettings,
        _: &[EncodedPaint],
        _: &dyn ImageResolver,
    ) {
        unreachable!("a recording has an open layer, which cannot be rasterized");
    }

    fn is_multi_threaded(&self) -> bool {
        false
    }
}

/// A filter layer of a recording.
#[derive(Debug)]
struct GraphLayer {
    filter: FilterData,
    /// Translation from root target coordinates to the layer's source space,
    /// where its commands are given.
    root_shift: (i64, i64),
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    steps: Vec<Step>,
    /// The filter layers nested directly in this one.
    children: Vec<usize>,
}

impl GraphLayer {
    /// Return the filter with its parameters in the grid of root target
    /// pixels `2^level` wide.
    fn filter_at(&self, level: u32) -> FilterData {
        let scale = 1.0 / f64::from(1_u32 << level);
        FilterData::new(
            self.filter.filter.clone(),
            Affine::scale(scale) * self.filter.transform,
        )
    }
}

/// A recorded command, or a nested filter layer rendered on its own.
#[derive(Clone, Copy, Debug)]
enum Step {
    Command(usize),
    /// The filter layer `layer`, pushed by the command `push`.
    Nested {
        layer: usize,
        push: usize,
    },
}

/// Add the filter layer whose commands are `range` and the filter layers
/// nested in it to `layers`, parents before their children, and return its index.
fn parse(
    commands: &[Command],
    range: core::ops::Range<usize>,
    filter: FilterData,
    root_shift: (i64, i64),
    clips: Vec<Clip>,
    layers: &mut Vec<GraphLayer>,
) -> usize {
    let index = layers.len();
    layers.push(GraphLayer {
        filter,
        root_shift,
        clips: Vec::new(),
        steps: Vec::new(),
        children: Vec::new(),
    });
    let mut active_clips = clips.clone();
    let mut steps = Vec::new();
    let mut children = Vec::new();
    let mut command = range.start;
    while command < range.end {
        if let Command::PushLayer(layer) = &commands[command]
            && let Some(filter) = &layer.filter
        {
            let end = matching_pop(commands, command);
            let shift = filter.source_shift();
            let child = parse(
                commands,
                command + 1..end,
                filter.clone(),
                (
                    root_shift.0 + i64::from(shift.0),
                    root_shift.1 + i64::from(shift.1),
                ),
                active_clips.clone(),
                layers,
            );
            children.push(child);
            steps.push(Step::Nested {
                layer: child,
                push: command,
            });
            // Clips pushed or popped inside the filter layer stay in effect after it.
            for (inner, nested) in commands.iter().enumerate().take(end).skip(command + 1) {
                if track_clips(nested, &mut active_clips) {
                    steps.push(Step::Command(inner));
                }
            }
            command = end + 1;
            continue;
        }
        track_clips(&commands[command], &mut active_clips);
        steps.push(Step::Command(command));
        command += 1;
    }
    let layer = &mut layers[index];
    layer.clips = clips;
    layer.steps = steps;
    layer.children = children;
    index
}

/// Apply `command` to `clips` and return whether it pushes or pops a clip.
fn track_clips(command: &Command, clips: &mut Vec<Clip>) -> bool {
    match command {
        Command::PushClip(clip) => clips.push(clip.clone()),
        Command::PopClip => {
            clips.pop();
        }
        _ => return false,
    }
    true
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

/// A recorded filter layer and the filter layers nested in it, planned and
/// replayed into windows.
#[derive(Debug)]
pub(crate) struct FilterGraph {
    /// One node for each layer and pixel grid its output is needed in, children
    /// after their parents. The first is the recorded layer in the root grid.
    nodes: Vec<Node>,
}

/// A filter layer with its output in one pixel grid.
#[derive(Debug)]
struct Node {
    plan: FilterPlan,
    /// Windows of the source in the output's grid.
    windows: Vec<Window>,
    /// Windows of the source the filter reads, if that is coarser than the output.
    coarse: Vec<Window>,
    /// Where the output has content, in the output's grid.
    extents: Vec<Bounds>,
}

/// The planned regions of a node, before its windows are replayed.
struct Planned {
    layer: usize,
    /// The pixel grid of the output: root target pixels `2^level` wide.
    level: u32,
    plan: FilterPlan,
    /// Regions of the source in the output's grid.
    regions: Regions,
    /// Regions of the source the filter reads, if that is coarser than the output.
    coarse: Regions,
}

/// Plan, from the parents down, the output regions of every layer in every
/// grid it is needed in. The first layer is needed inside `view` in the root grid.
fn plan(layers: &[GraphLayer], view: Bounds) -> Vec<Planned> {
    let mut needed: Vec<BTreeMap<u32, Regions>> = layers.iter().map(|_| BTreeMap::new()).collect();
    needed[0].entry(0).or_default().insert(view);
    let mut planned = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        for (level, outputs) in core::mem::take(&mut needed[index]) {
            let plan = FilterPlan::new(layer.filter_at(level), &outputs);
            let (mut regions, coarse) = plan.source_regions();
            let coarse = if plan.reduction() == 0 {
                coarse.iter().for_each(|region| regions.insert(region));
                Regions::default()
            } else {
                coarse
            };
            for &child in &layer.children {
                for (level, regions) in [(level, &regions), (level + plan.reduction(), &coarse)] {
                    if regions.len() != 0 {
                        let outputs = needed[child].entry(level).or_default();
                        regions.iter().for_each(|region| outputs.insert(region));
                    }
                }
            }
            planned.push(Planned {
                layer: index,
                level,
                plan,
                regions,
                coarse,
            });
        }
    }
    planned
}

impl FilterGraph {
    fn new(commands: &[Command], layers: &[GraphLayer], view: Bounds, simd_level: Level) -> Self {
        let planned = plan(layers, view);
        let index: BTreeMap<(usize, u32), usize> = planned
            .iter()
            .enumerate()
            .map(|(node, planned)| ((planned.layer, planned.level), node))
            .collect();
        // Replay children first, so that a parent's windows know where the
        // children they composite have content.
        let mut nodes: Vec<Option<Node>> = planned.iter().map(|_| None).collect();
        for (node, planned) in planned.into_iter().enumerate().rev() {
            let replay = Replay {
                layer: &layers[planned.layer],
                commands,
                layers,
                index: &index,
                nodes: &nodes,
                simd_level,
            };
            let windows = replay.windows(&planned.regions, planned.level);
            let coarse = replay.windows(&planned.coarse, planned.level + planned.plan.reduction());
            let contents = |windows: &[Window]| -> Vec<Bounds> {
                windows.iter().map(Window::content).collect()
            };
            let kept = contents(&windows);
            let filtered = if planned.plan.reduction() == 0 {
                kept.clone()
            } else {
                contents(&coarse)
            };
            let extents = planned.plan.extents(&kept, &filtered);
            nodes[node] = Some(Node {
                plan: planned.plan,
                windows,
                coarse,
                extents,
            });
        }
        Self {
            nodes: nodes.into_iter().map(Option::unwrap).collect(),
        }
    }

    /// Return where the output has content in the root grid.
    fn extents(&self) -> &[Bounds] {
        &self.nodes[0].extents
    }

    /// Return where the visible part of the output is composited in `view`.
    pub(crate) fn placement_in(&self, view: Bounds) -> RectU16 {
        self.extents()
            .iter()
            .fold(Bounds::EMPTY, |placement, extent| {
                placement.union(extent.intersect(view))
            })
            .snap_to_tiles()
            .to_rect_u16((0, 0))
    }

    /// Render the layer and the filters nested in it, and return its output in the root grid.
    pub(crate) fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> Vec<PlacedPixmap> {
        let mut outputs: Vec<Vec<PlacedPixmap>> = self.nodes.iter().map(|_| Vec::new()).collect();
        let mut scratch = ScratchBuffer::new();
        for (index, node) in self.nodes.iter().enumerate().rev() {
            let render = |windows: &[Window]| -> Vec<PlacedPixmap> {
                windows
                    .iter()
                    .filter_map(|window| {
                        window.render::<S, F>(simd, encoded_paints, image_resolver, &outputs)
                    })
                    .collect()
            };
            let kept = render(&node.windows);
            let filtered = if node.plan.reduction() == 0 {
                None
            } else {
                Some(render(&node.coarse))
            };
            outputs[index] = node.plan.render(
                &kept,
                filtered.as_deref().unwrap_or(&kept),
                &mut scratch,
                &mut |pixmap, filter, scratch| F::filter_layer(pixmap, filter, scratch),
            );
        }
        core::mem::take(&mut outputs[0])
    }
}

/// Return a copy of the pixels of `pieces` inside `area`, transparent where they do not cover it.
pub(crate) fn crop(pieces: &[PlacedPixmap], area: Bounds) -> PlacedPixmap {
    let mut crop = PlacedPixmap::new(area);
    for piece in pieces {
        crop.copy_from(piece, area);
    }
    crop
}

/// What replaying the windows of one layer needs.
struct Replay<'a> {
    layer: &'a GraphLayer,
    commands: &'a [Command],
    layers: &'a [GraphLayer],
    /// The node of each layer and grid.
    index: &'a BTreeMap<(usize, u32), usize>,
    /// The nodes replayed so far, which include the children of this layer.
    nodes: &'a [Option<Node>],
    simd_level: Level,
}

impl Replay<'_> {
    /// Replay the layer into windows covering `regions` in the grid of root
    /// target pixels `2^level` wide.
    fn windows(&self, regions: &Regions, level: u32) -> Vec<Window> {
        let mut masks = Masks::new(level);
        regions
            .iter()
            .flat_map(window_grid)
            .map(|window| self.window(window, level, &mut masks))
            .collect()
    }

    fn window(&self, window: Bounds, level: u32, masks: &mut Masks) -> Window {
        let origin = (window.x0, window.y0);
        let to_grid = Affine::translate((-origin.0 as f64, -origin.1 as f64))
            * Affine::scale(1.0 / f64::from(1_u32 << level));
        let shift = self.layer.root_shift;
        let to_window = to_grid * Affine::translate((-shift.0 as f64, -shift.1 as f64));
        let dispatcher = SingleThreadedDispatcher::window(
            u16::try_from(window.width()).unwrap(),
            u16::try_from(window.height()).unwrap(),
            self.simd_level,
            origin,
            |target| {
                for clip in &self.layer.clips {
                    clip.push_into(target, to_grid);
                }
                for step in &self.layer.steps {
                    match *step {
                        Step::Command(command) => {
                            apply(target, &self.commands[command], to_window, to_grid, masks);
                        }
                        Step::Nested { layer, push } => {
                            let Command::PushLayer(pushed) = &self.commands[push] else {
                                unreachable!("a nested layer starts with its push");
                            };
                            let node = self.index[&(layer, level)];
                            let extents = &self.nodes[node].as_ref().unwrap().extents;
                            inject(
                                target,
                                pushed,
                                &self.layers[layer].filter,
                                node,
                                extents,
                                window,
                                to_window,
                                masks,
                            );
                        }
                    }
                }
            },
        );
        Window {
            origin,
            level,
            root_shift: shift,
            dispatcher,
        }
    }
}

/// Masks sampled in the grid of root target pixels `2^level` wide.
struct Masks {
    level: u32,
    /// Masks of the root target with their copies in this grid.
    reduced: Vec<(Mask, Mask)>,
}

impl Masks {
    fn new(level: u32) -> Self {
        Self {
            level,
            reduced: Vec::new(),
        }
    }

    /// Return `mask` in this grid, where each pixel averages the pixels of the
    /// root target mask it covers.
    fn get(&mut self, mask: &Mask) -> Mask {
        if self.level == 0 {
            return mask.clone();
        }
        if let Some((_, reduced)) = self.reduced.iter().find(|(root, _)| root == mask) {
            return reduced.clone();
        }
        let reduced = reduce_mask(mask, self.level);
        self.reduced.push((mask.clone(), reduced.clone()));
        reduced
    }
}

/// Return `mask` with each pixel averaging a square of `2^levels` pixels of
/// it; pixels outside the mask count as transparent, like when it is sampled.
fn reduce_mask(mask: &Mask, levels: u32) -> Mask {
    let step = 1_u32 << levels;
    let width = u32::from(mask.width()).div_ceil(step);
    let height = u32::from(mask.height()).div_ceil(step);
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0_u64;
            for sy in y * step..((y + 1) * step).min(u32::from(mask.height())) {
                for sx in x * step..((x + 1) * step).min(u32::from(mask.width())) {
                    sum += u64::from(mask.sample(sx as u16, sy as u16));
                }
            }
            let count = u64::from(step) * u64::from(step);
            data.push(((sum + count / 2) / count) as u8);
        }
    }
    Mask::from_parts(data, width as u16, height as u16)
}

/// A window of a layer's source, replayed into its own dispatcher.
#[derive(Debug)]
struct Window {
    /// The window's origin, in the grid of root target pixels `2^level` wide.
    origin: (i64, i64),
    level: u32,
    /// Translation from root target coordinates to the layer's source space.
    root_shift: (i64, i64),
    dispatcher: SingleThreadedDispatcher,
}

impl Window {
    /// Return the bounds of the window's content in its grid.
    fn content(&self) -> Bounds {
        self.dispatcher.window_content()
    }

    /// Render the window content, or return `None` if it is empty. `nested`
    /// holds the outputs of the nodes of the graph rendered so far.
    fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        nested: &[Vec<PlacedPixmap>],
    ) -> Option<PlacedPixmap> {
        let encoded_paints = paints_at(encoded_paints, self.to_source());
        self.dispatcher
            .render_window::<S, F>(simd, &encoded_paints, image_resolver, nested)
    }

    /// Return the transform from the window's pixels to the layer's source space.
    fn to_source(&self) -> Affine {
        let scale = f64::from(1_u32 << self.level);
        Affine::translate((self.root_shift.0 as f64, self.root_shift.1 as f64))
            * Affine::scale(scale)
            * Affine::translate((self.origin.0 as f64, self.origin.1 as f64))
    }
}

/// Return copies of `paints` for sampling at window pixels mapped to the
/// paints' space by `to_source`, which scales by a power of two.
fn paints_at(paints: &[EncodedPaint], to_source: Affine) -> Vec<EncodedPaint> {
    let scale = to_source.as_coeffs()[0];
    paints
        .iter()
        .map(|paint| {
            let mut paint = paint.clone();
            let (transform, x_advance, y_advance) = match &mut paint {
                EncodedPaint::Gradient(gradient) => (
                    &mut gradient.transform,
                    &mut gradient.x_advance,
                    &mut gradient.y_advance,
                ),
                EncodedPaint::Image(image) => (
                    &mut image.transform,
                    &mut image.x_advance,
                    &mut image.y_advance,
                ),
                EncodedPaint::BlurredRoundedRect(rect) => (
                    &mut rect.transform,
                    &mut rect.x_advance,
                    &mut rect.y_advance,
                ),
            };
            *transform *= to_source;
            *x_advance *= scale;
            *y_advance *= scale;
            paint
        })
        .collect()
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

/// Composite the part of a nested filter's output that falls into `window`.
fn inject(
    target: &mut SingleThreadedDispatcher,
    layer: &Layer,
    filter: &FilterData,
    node: usize,
    extents: &[Bounds],
    window: Bounds,
    to_window: Affine,
    masks: &mut Masks,
) {
    let dest = extents
        .iter()
        .fold(Bounds::EMPTY, |dest, extent| {
            dest.union(extent.intersect(window))
        })
        .snap_to_tiles()
        .to_rect_u16((window.x0, window.y0));
    let mask = layer.mask.as_ref().map(|mask| masks.get(mask));
    target.push_rendered_filter_layer(layer, mask, to_window, filter.clone(), dest, node);
}

fn apply(
    target: &mut SingleThreadedDispatcher,
    command: &Command,
    to_window: Affine,
    to_grid: Affine,
    masks: &mut Masks,
) {
    let mut mask = |mask: &Option<Mask>| mask.as_ref().map(|mask| masks.get(mask));
    match command {
        Command::FillPath {
            path,
            fill_rule,
            transform,
            paint,
            blend_mode,
            aliasing_threshold,
            mask: path_mask,
        } => target.fill_path(
            path,
            *fill_rule,
            to_window * *transform,
            paint.clone(),
            *blend_mode,
            *aliasing_threshold,
            mask(path_mask),
        ),
        Command::StrokePath {
            path,
            stroke,
            transform,
            paint,
            blend_mode,
            aliasing_threshold,
            mask: path_mask,
        } => target.stroke_path(
            path,
            stroke,
            to_window * *transform,
            paint.clone(),
            *blend_mode,
            *aliasing_threshold,
            mask(path_mask),
        ),
        Command::FillRect {
            rect,
            paint,
            blend_mode,
            mask: rect_mask,
        } => target.fill_rect_fast(
            &to_window.transform_rect_bbox(*rect),
            paint.clone(),
            *blend_mode,
            mask(rect_mask),
        ),
        Command::PushLayer(layer) => target.push_layer(
            layer.clip_path.as_ref(),
            layer.fill_rule,
            to_window * layer.clip_transform,
            layer.blend_mode,
            layer.opacity,
            layer.aliasing_threshold,
            mask(&layer.mask),
            None,
        ),
        Command::PopLayer => target.pop_layer(),
        Command::PushClip(clip) => clip.push_into(target, to_grid),
        Command::PopClip => target.pop_clip(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::Vec2;
    use alloc::vec;
    use vello_common::color::palette::css::BLUE;
    use vello_common::paint::PremulColor;

    /// Record `filters`, each nested in the one before, around `rects` in root
    /// target coordinates, as `RenderContext` would, and plan them in `view`.
    fn graph(view: Bounds, filters: &[FilterPrimitive], rects: &[Rect]) -> FilterGraph {
        let mut shift = (0.0, 0.0);
        let mut filter_data = |primitive: &FilterPrimitive| {
            let data = FilterData::new(
                Filter::from_primitive(primitive.clone()),
                Affine::translate(shift),
            );
            let (dx, dy) = data.source_shift();
            shift = (shift.0 + f64::from(dx), shift.1 + f64::from(dy));
            data
        };
        let mut recording = Recording::new(filter_data(&filters[0]), view, Vec::new());
        for primitive in &filters[1..] {
            recording.push_layer(
                None,
                Fill::NonZero,
                Affine::IDENTITY,
                BlendMode::default(),
                1.0,
                None,
                None,
                Some(filter_data(primitive)),
            );
        }
        for rect in rects {
            recording.fill_rect_fast(
                &(*rect + Vec2::from(shift)),
                Paint::Solid(PremulColor::from_alpha_color(BLUE)),
                BlendMode::default(),
                None,
            );
        }
        for _ in 0..filters.len() {
            recording.pop_layer();
        }
        recording.finish(Level::fallback())
    }

    /// What rendering a graph takes.
    #[derive(Debug, PartialEq)]
    struct Work {
        nodes: usize,
        windows: usize,
        /// The pixels of the windows the source is rasterized in.
        window_pixels: i64,
        strips: usize,
        /// The pixels of the pixmaps the filters run on.
        pixmap_pixels: i64,
    }

    fn work(graph: &FilterGraph) -> Work {
        let windows = || {
            graph
                .nodes
                .iter()
                .flat_map(|node| node.windows.iter().chain(&node.coarse))
        };
        let pixmap_pixels = graph
            .nodes
            .iter()
            .flat_map(|node| {
                let kept: Vec<_> = node.windows.iter().map(Window::content).collect();
                let filtered = if node.plan.reduction() == 0 {
                    kept.clone()
                } else {
                    node.coarse.iter().map(Window::content).collect()
                };
                node.plan.pixmaps(&kept, &filtered)
            })
            .map(Bounds::area)
            .sum();
        Work {
            nodes: graph.nodes.len(),
            windows: windows().count(),
            window_pixels: windows()
                .map(|window| {
                    let (width, height) = window.dispatcher.size();
                    i64::from(width) * i64::from(height)
                })
                .sum(),
            strips: windows()
                .map(|window| window.dispatcher.strip_count())
                .sum(),
            pixmap_pixels,
        }
    }

    const VIEW: Bounds = Bounds {
        x0: 0,
        y0: 0,
        x1: 64,
        y1: 64,
    };

    #[test]
    fn nested_shadows_read_each_source_region_once() {
        const STEP: f64 = 65_536.0;
        let shadow = FilterPrimitive::DropShadow {
            dx: STEP as f32,
            dy: 0.0,
            std_deviation: 1.0,
            color: BLUE,
            edge_mode: EdgeMode::None,
        };
        for depth in 1..=16_usize {
            let x = 16.0 - STEP * depth as f64;
            let graph = graph(
                VIEW,
                &vec![shadow.clone(); depth],
                &[Rect::new(x, 16.0, x + 32.0, 48.0)],
            );
            let work = work(&graph);
            // The layer nested `n` deep reads the view and `n + 1` shadowed copies.
            assert_eq!(work.nodes, depth, "depth {depth}");
            assert_eq!(work.windows, depth * (depth + 3) / 2, "depth {depth}");
        }
    }

    #[test]
    fn moved_source_is_rasterized_only_where_it_is_read() {
        let blur = FilterPrimitive::GaussianBlur {
            std_deviation: 4.0,
            edge_mode: EdgeMode::None,
        };
        let work_at = |distance: f64| {
            let offset = FilterPrimitive::Offset {
                dx: -distance as f32,
                dy: -distance as f32,
            };
            // A band from the origin to the source that is read, and a square in the nested blur.
            let band = Rect::new(distance + 4.0, 0.0, distance + 12.0, distance + 64.0);
            let square = Rect::new(
                distance + 24.0,
                distance + 24.0,
                distance + 48.0,
                distance + 48.0,
            );
            work(&graph(VIEW, &[offset, blur.clone()], &[band, square]))
        };
        // The first distance keeps the source inside the `u16` domain.
        let near = work_at(96.0);
        assert!(near.strips > 0);
        for distance in [4_096.0, 60_000.0, 262_144.0, 16_777_216.0] {
            assert_eq!(work_at(distance), near, "distance {distance}");
        }
    }

    #[test]
    fn wide_blur_reads_a_bounded_source() {
        let view = Bounds::new(0, 0, 12, 12);
        for std_deviation in [4.0_f32, 40.0, 400.0, 4_000.0, 20_000.0] {
            let blur = FilterPrimitive::GaussianBlur {
                std_deviation,
                edge_mode: EdgeMode::None,
            };
            let reach = 4.0 * f64::from(std_deviation);
            let square = Rect::new(-reach, -reach, 12.0 + reach, 12.0 + reach);
            let work = work(&graph(view, &[blur], &[square]));
            assert!(
                work.window_pixels <= 1 << 20 && work.pixmap_pixels <= 1 << 20,
                "sigma {std_deviation}: {work:?}"
            );
        }
    }
}
