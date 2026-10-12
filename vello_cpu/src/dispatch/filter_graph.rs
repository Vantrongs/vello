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
//! The commands are then replayed into windows that cover those regions.
//! Strips, tiles and fine rasterization address pixels with `u16`, so a
//! window is at most 65532 pixels wide; regions and windows themselves are
//! placed in coordinates of the root target wide enough for any copy that
//! filters move away and back, see [`Coord`].
//!
//! Nested filters are rendered first and handed to the windows of their parent
//! as finished pixmaps, so every filter sees all of the source it reads and
//! the result does not depend on where window edges fall. A wide blur reads
//! its source decimated from tiles rendered at full resolution, and the
//! filters nested in that source are streamed tile by tile, see
//! [`Rendering::decimate`].

use super::Dispatcher;
use super::single_threaded::SingleThreadedDispatcher;
use crate::RasterizerSettings;
use crate::filter::context::ScratchBuffer;
use crate::filter::placed::{Bounds, Coord, Index, Pieces, PlacedPixmap, Position, Regions};
use crate::filter::plan::{Blurred, FilterPlan, Reach};
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Join, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use vello_common::clip::PushedClip;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::FilterData;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;

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

    /// Return a rectangle in root target coordinates outside of which the clip
    /// hides everything.
    fn bounds(&self) -> Rect {
        match self {
            Self::Path {
                path, transform, ..
            } => transform.transform_rect_bbox(path.control_box()),
            Self::Rect(rect) => *rect,
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
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    commands: Vec<Command>,
    /// The number of open layers, the recorded layer included.
    depth: usize,
}

impl Recording {
    /// Start recording a filter layer whose output is visible in `view`.
    pub(crate) fn new(filter: FilterData, view: Bounds, clips: Vec<Clip>) -> Self {
        Self {
            filter,
            view,
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
            self.clips,
            &mut layers,
        );
        FilterGraph::new(self.commands, layers, self.view, level)
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
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    steps: Vec<Step>,
    /// The filter layers nested directly in this one.
    children: Vec<Child>,
    /// A rectangle in root target coordinates that holds everything the
    /// layer's own commands draw, if they draw anything.
    drawn: Option<Rect>,
}

/// A filter layer nested directly in another.
#[derive(Clone, Copy, Debug)]
struct Child {
    layer: usize,
    /// A rectangle in root target coordinates outside of which the clip
    /// paths of the layers it is composited through hide it.
    clip: Rect,
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
    clips: Vec<Clip>,
    layers: &mut Vec<GraphLayer>,
) -> usize {
    let index = layers.len();
    layers.push(GraphLayer {
        filter,
        clips: Vec::new(),
        steps: Vec::new(),
        children: Vec::new(),
        drawn: None,
    });
    let mut active_clips = clips.clone();
    let mut clip_bounds = ClipBounds::new(&clips);
    let mut steps = Vec::new();
    let mut children = Vec::new();
    let mut drawn: Option<Rect> = None;
    let mut command = range.start;
    while command < range.end {
        if let Command::PushLayer(layer) = &commands[command]
            && let Some(filter) = &layer.filter
        {
            let end = matching_pop(commands, command);
            let clip = layer.clip_bounds().intersect(clip_bounds.composited());
            let child = parse(
                commands,
                command + 1..end,
                filter.clone(),
                active_clips.clone(),
                layers,
            );
            children.push(Child { layer: child, clip });
            steps.push(Step::Nested {
                layer: child,
                push: command,
            });
            // Clips pushed or popped inside the filter layer stay in effect after it.
            for (inner, nested) in commands.iter().enumerate().take(end).skip(command + 1) {
                if track_clips(nested, &mut active_clips) {
                    clip_bounds.track(nested);
                    steps.push(Step::Command(inner));
                }
            }
            command = end + 1;
            continue;
        }
        track_clips(&commands[command], &mut active_clips);
        clip_bounds.track(&commands[command]);
        if let Some(bounds) = commands[command].bounds() {
            let bounds = bounds.intersect(clip_bounds.drawn());
            drawn = Some(drawn.map_or(bounds, |drawn| drawn.union(bounds)));
        }
        steps.push(Step::Command(command));
        command += 1;
    }
    let layer = &mut layers[index];
    layer.clips = clips;
    layer.steps = steps;
    layer.children = children;
    layer.drawn = drawn;
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

/// A rectangle in root target coordinates as wide as everything.
const UNBOUNDED: Rect = Rect::new(
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::INFINITY,
);

/// Return `rect`, or [`UNBOUNDED`] if it is not finite.
fn finite_or_unbounded(rect: Rect) -> Rect {
    if [rect.x0, rect.y0, rect.x1, rect.y1]
        .iter()
        .all(|value| value.is_finite())
    {
        rect
    } else {
        UNBOUNDED
    }
}

/// Rectangles outside of which the clips in effect hide everything: those of
/// clips, which are pushed and popped on their own, and those of the clip
/// paths of layers. Each stack starts with everything, which is never popped.
struct ClipBounds {
    clips: Vec<Rect>,
    layers: Vec<Rect>,
}

impl ClipBounds {
    /// Start with `clips` in effect, which can be popped like the others.
    fn new(clips: &[Clip]) -> Self {
        let mut bounds = Self {
            clips: vec![UNBOUNDED],
            layers: vec![UNBOUNDED],
        };
        for clip in clips {
            bounds.push_clip(clip);
        }
        bounds
    }

    fn push_clip(&mut self, clip: &Clip) {
        let bounds = finite_or_unbounded(clip.bounds());
        self.clips
            .push(self.clips.last().unwrap().intersect(bounds));
    }

    /// Return the bounds of what is drawn.
    fn drawn(&self) -> Rect {
        self.clips.last().unwrap().intersect(self.composited())
    }

    /// Return the bounds of the output of a filter layer, which clips pushed
    /// on their own leave as it is.
    fn composited(&self) -> Rect {
        *self.layers.last().unwrap()
    }

    fn track(&mut self, command: &Command) {
        match command {
            Command::PushClip(clip) => self.push_clip(clip),
            Command::PopClip if self.clips.len() > 1 => {
                self.clips.pop();
            }
            Command::PushLayer(layer) => {
                let bounds = layer.clip_bounds();
                self.layers
                    .push(self.layers.last().unwrap().intersect(bounds));
            }
            Command::PopLayer if self.layers.len() > 1 => {
                self.layers.pop();
            }
            _ => {}
        }
    }
}

impl Layer {
    /// Return a rectangle in root target coordinates outside of which the
    /// layer's clip path hides it.
    fn clip_bounds(&self) -> Rect {
        self.clip_path.as_ref().map_or(UNBOUNDED, |path| {
            finite_or_unbounded(self.clip_transform.transform_rect_bbox(path.control_box()))
        })
    }
}

impl Command {
    /// Return a rectangle in root target coordinates that holds everything the
    /// command draws, or `None` if it draws nothing itself.
    fn bounds(&self) -> Option<Rect> {
        let bounds = match self {
            Self::FillPath {
                path, transform, ..
            } => transform.transform_rect_bbox(path.control_box()),
            Self::StrokePath {
                path,
                stroke,
                transform,
                ..
            } => {
                // A miter reaches the miter limit times the half width from
                // the path, a square cap the diagonal of the half width.
                let reach = if stroke.join == Join::Miter {
                    stroke.miter_limit.max(core::f64::consts::SQRT_2)
                } else {
                    core::f64::consts::SQRT_2
                } * stroke.width.abs()
                    / 2.0;
                // At least the largest stretch of the transform.
                let [a, b, c, d, _, _] = transform.as_coeffs();
                let device = reach * (a.abs() + b.abs() + c.abs() + d.abs());
                transform
                    .transform_rect_bbox(path.control_box())
                    .inflate(device, device)
            }
            Self::FillRect { rect, .. } => *rect,
            _ => return None,
        };
        // Covers what anti-aliasing and rounding add around the geometry.
        Some(finite_or_unbounded(bounds.inflate(1.0, 1.0)))
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

/// A recorded filter layer and the filter layers nested in it, planned and
/// replayed into windows.
#[derive(Debug)]
pub(crate) struct FilterGraph {
    commands: Vec<Command>,
    layers: Vec<GraphLayer>,
    /// One node for each layer, children after their parents. The first is
    /// the recorded layer.
    nodes: Vec<Node>,
    simd_level: Level,
}

/// A filter layer with its output planned.
///
/// The output is needed where it is composited into the windows of its parent
/// or into the view, which are replayed in advance, and where the source of a
/// blur that reads a coarser source is decimated, which streams the source of
/// its nested filter layers tile by tile, see [`Rendering::decimate`].
#[derive(Debug)]
struct Node {
    /// The output that is composited into windows replayed in advance.
    plan: FilterPlan,
    /// Windows of the source of `plan`.
    windows: Vec<Window>,
    /// Where the output of `plan` has content.
    extents: Index,
    /// The output that is read where a coarser source is decimated.
    streamed: FilterPlan,
    /// Where the source of `streamed` can have content, and where the source
    /// its filter reads can, in that source's grid.
    content: Index,
    filtered: Index,
    /// Regions of the source the filter reads if that is coarser than the
    /// output, each pixel the average of the source pixels it covers.
    decimated: Vec<Bounds>,
    /// Where the source of `decimated` can have content at full resolution.
    decimated_content: Regions,
}

/// The planned regions of a node, before its windows are replayed.
struct Planned {
    plan: FilterPlan,
    streamed: FilterPlan,
    /// Regions of the source to replay into windows in advance.
    regions: Regions,
    /// Where the source the output keeps can have content.
    kept_content: Regions,
    /// Where the source the filter reads can have content, in its grid.
    filtered_content: Regions,
    decimated: Regions,
    decimated_content: Regions,
}

/// Plan the output regions of every layer. The first layer is needed inside `view`.
///
/// A region of a source is rendered only where that source has content, so
/// that the copies a drop shadow that keeps its source makes of every copy
/// nested in it are read only where one is visible. Where the layers have
/// content is first found from the children up, inside the regions the
/// parents can read at most, so that it too stays within what is visible.
fn plan(layers: &[GraphLayer], view: Bounds) -> Vec<Planned> {
    let count = layers.len();
    // From the parents down: how far each layer is needed.
    let mut hulls = vec![Bounds::EMPTY; count];
    hulls[0] = view;
    let mut reaches = Vec::with_capacity(count);
    for (index, layer) in layers.iter().enumerate() {
        let reach = Reach::new(&layer.filter);
        let (kept, filtered) = reach.reads(hulls[index]);
        let source = kept.union(filtered.refine(reach.reduction()));
        for child in &layer.children {
            hulls[child.layer] = hulls[child.layer].union(source);
        }
        reaches.push(reach);
    }
    // From the children up: where each layer has content.
    let mut contents = vec![Regions::default(); count];
    let mut kept_contents = vec![Regions::default(); count];
    let mut filtered_contents = vec![Regions::default(); count];
    let mut read_contents = vec![Regions::default(); count];
    for index in (0..count).rev() {
        let (reach, hull) = (&reaches[index], hulls[index]);
        let reduction = reach.reduction();
        let (kept_area, filtered_area) = reach.reads(hull);
        let content_in = |area| source_content(&layers[index], area, &contents);
        let (kept, read) = if reduction == 0 {
            let content = content_in(kept_area.union(filtered_area));
            (content.clone(), content)
        } else {
            (
                content_in(kept_area),
                content_in(filtered_area.refine(reduction)),
            )
        };
        let mut filtered = Regions::default();
        read.iter()
            .for_each(|region| filtered.insert(region.coarsen(reduction)));
        let mut content = Regions::default();
        for region in filtered.iter() {
            content.insert(reach.writes(region).intersect(hull));
        }
        if reach.keeps_source() {
            for region in kept.iter() {
                content.insert(region);
            }
        }
        contents[index] = content;
        kept_contents[index] = kept;
        filtered_contents[index] = filtered;
        read_contents[index] = read;
    }
    // From the parents down: the regions of each layer that are needed.
    let mut shown = vec![Regions::default(); count];
    shown[0].insert(view);
    let mut streamed = vec![Regions::default(); count];
    let mut planned = Vec::with_capacity(count);
    for (index, layer) in layers.iter().enumerate() {
        let reduction = reaches[index].reduction();
        let plan = FilterPlan::new(reaches[index].clone(), &core::mem::take(&mut shown[index]));
        let stream = FilterPlan::new(
            reaches[index].clone(),
            &core::mem::take(&mut streamed[index]),
        );
        let (mut windows, filtered) = plan.source_regions();
        let (mut streams, stream_filtered) = stream.source_regions();
        let mut decimated = Regions::default();
        if reduction == 0 {
            filtered.iter().for_each(|region| windows.insert(region));
            stream_filtered
                .iter()
                .for_each(|region| streams.insert(region));
        } else {
            let mut coarse = filtered;
            stream_filtered
                .iter()
                .for_each(|region| coarse.insert(region));
            decimated = intersection(&coarse, &filtered_contents[index]);
            decimated
                .iter()
                .for_each(|region| streams.insert(region.refine(reduction)));
        }
        for child in &layer.children {
            let clip = Bounds::covering(child.clip, 1.0);
            for content in contents[child.layer].iter() {
                let content = content.intersect(clip);
                for (sources, needed) in [(&windows, &mut shown), (&streams, &mut streamed)] {
                    for source in sources.touching(content) {
                        needed[child.layer].insert(source.intersect(content));
                    }
                }
            }
        }
        planned.push(Planned {
            plan,
            streamed: stream,
            regions: intersection(&windows, &kept_contents[index]),
            kept_content: core::mem::take(&mut kept_contents[index]),
            filtered_content: core::mem::take(&mut filtered_contents[index]),
            decimated,
            decimated_content: core::mem::take(&mut read_contents[index]),
        });
    }
    planned
}

/// Return where the source of `layer` has content inside `area`: its own
/// drawing and the content of its children, inside their clips.
fn source_content(layer: &GraphLayer, area: Bounds, contents: &[Regions]) -> Regions {
    let mut content = Regions::default();
    if let Some(drawn) = layer.drawn {
        content.insert(Bounds::covering(drawn, 1.0).intersect(area));
    }
    for child in &layer.children {
        let clip = Bounds::covering(child.clip, 1.0).intersect(area);
        for region in contents[child.layer].iter() {
            content.insert(region.intersect(clip));
        }
    }
    content
}

/// Return the parts of `regions` inside `content`.
fn intersection(regions: &Regions, content: &Regions) -> Regions {
    let mut parts = Regions::default();
    for region in regions.iter() {
        for content in content.touching(region) {
            parts.insert(region.intersect(content));
        }
    }
    parts
}

impl FilterGraph {
    fn new(
        commands: Vec<Command>,
        layers: Vec<GraphLayer>,
        view: Bounds,
        simd_level: Level,
    ) -> Self {
        let planned = plan(&layers, view);
        // Replay children first, so that a parent's windows know where the
        // children they composite have content.
        let mut extents = vec![Index::default(); layers.len()];
        let mut nodes = Vec::with_capacity(layers.len());
        for (index, planned) in planned.into_iter().enumerate().rev() {
            let replay = Replay {
                layer: &layers[index],
                commands: &commands,
                layers: &layers,
                extents: &extents,
                simd_level,
            };
            let windows = replay.windows(&planned.regions);
            let kept = Index::new(windows.iter().map(Window::content));
            let reduced = planned.plan.reduction() > 0;
            let filtered = if reduced {
                Index::new(planned.decimated.iter())
            } else {
                kept.clone()
            };
            extents[index] = Index::new(planned.plan.extents(&kept, &filtered));
            let content = Index::new(planned.kept_content.iter());
            nodes.push(Node {
                plan: planned.plan,
                windows,
                extents: Index::default(),
                streamed: planned.streamed,
                filtered: if reduced {
                    filtered
                } else {
                    Index::new(planned.filtered_content.iter())
                },
                content,
                decimated: planned.decimated.iter().collect(),
                decimated_content: planned.decimated_content,
            });
        }
        nodes.reverse();
        for (node, extents) in nodes.iter_mut().zip(extents) {
            node.extents = extents;
        }
        Self {
            commands,
            layers,
            nodes,
            simd_level,
        }
    }

    /// Return where the output has content in the root grid.
    fn extents(&self) -> &Index {
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
            .to_rect_u16((Coord::ZERO, Coord::ZERO))
    }

    /// Render the layer and the filters nested in it, and return its output in the root grid.
    pub(crate) fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> Pieces {
        let mut rendering = Rendering::<S, F> {
            graph: self,
            simd,
            encoded_paints,
            image_resolver,
            decimated: self.nodes.iter().map(|_| Pieces::default()).collect(),
            blurred: BTreeMap::new(),
            scratch: ScratchBuffer::new(),
            kernel: PhantomData,
        };
        // Children first: decimating a source streams the filters nested in it.
        for index in (0..self.nodes.len()).rev() {
            if !self.nodes[index].decimated.is_empty() {
                rendering.decimated[index] = rendering.decimate(index);
            }
        }
        let mut outputs: Vec<Pieces> = self.nodes.iter().map(|_| Pieces::default()).collect();
        for (index, node) in self.nodes.iter().enumerate().rev() {
            let kept = Pieces::new(
                node.windows
                    .iter()
                    .filter_map(|window| {
                        window.render::<S, F>(simd, encoded_paints, image_resolver, &outputs)
                    })
                    .collect(),
            );
            let filtered = if node.plan.reduction() == 0 {
                &kept
            } else {
                &rendering.decimated[index]
            };
            outputs[index] = Pieces::new(node.plan.render(
                &kept,
                filtered,
                &mut rendering.scratch,
                &mut |pixmap, filter, scratch| F::filter_layer(pixmap, filter, scratch),
            ));
        }
        core::mem::take(&mut outputs[0])
    }
}

/// The tiles a source is rendered in at full resolution while it is
/// decimated are `2^TILE_LEVELS` pixels wide.
const TILE_LEVELS: u32 = 10;

/// The state of rendering a filter graph.
struct Rendering<'a, S: Simd, F> {
    graph: &'a FilterGraph,
    simd: S,
    encoded_paints: &'a [EncodedPaint],
    image_resolver: &'a dyn ImageResolver,
    /// The decimated source of each node that reads one.
    decimated: Vec<Pieces>,
    /// The pixmaps that blurs reading a decimated source ran on, by node and
    /// region of their streamed output.
    blurred: BTreeMap<(usize, usize), Blurred>,
    scratch: ScratchBuffer,
    kernel: PhantomData<F>,
}

impl<S: Simd, F: FineKernel<S>> Rendering<'_, S, F> {
    /// Return the source the filter of `node` reads in the regions
    /// [`Node::decimated`], with each pixel the average of the pixels of the
    /// full-resolution source it covers.
    ///
    /// The source is rendered at full resolution in tiles that line up with
    /// the coarse pixels and only where it has content, so memory stays within
    /// a tile and the coarse regions. The work is that of rendering the source
    /// the visible output depends on at full resolution, which grows with the
    /// square of the blur's width.
    fn decimate(&mut self, node: usize) -> Pieces {
        let graph = self.graph;
        let graph_node = &graph.nodes[node];
        let levels = graph_node.plan.reduction();
        let content = &graph_node.decimated_content;
        let mut pieces = Vec::with_capacity(graph_node.decimated.len());
        for &region in &graph_node.decimated {
            let mut coarse = PlacedPixmap::new(region);
            if levels <= TILE_LEVELS {
                // Each tile holds whole squares of the coarse pixels.
                for tile in content_tiles(region.refine(levels), content) {
                    let source = self.source(node, &Regions::of(tile));
                    if !source.is_empty() {
                        coarse.average(&Pieces::new(source).crop(tile), levels);
                    }
                }
            } else {
                let (width, height) = region.size();
                for (index, (y, x)) in (0..height)
                    .flat_map(|y| (0..width).map(move |x| (y, x)))
                    .enumerate()
                {
                    let (x, y) = (
                        region.x0 + i64::from(x).into(),
                        region.y0 + i64::from(y).into(),
                    );
                    let square = Bounds::new(x, y, x + Coord::from(1), y + Coord::from(1));
                    let mut sums = [0_u128; 4];
                    for tile in content_tiles(square.refine(levels), content) {
                        for piece in self.source(node, &Regions::of(tile)) {
                            piece.add_to(&mut sums);
                        }
                    }
                    coarse.set_average(index, sums, levels);
                }
            }
            pieces.push(coarse);
        }
        Pieces::new(pieces)
    }

    /// Render the source of `node` inside `regions` at full resolution, with
    /// the nested filter layers streamed.
    fn source(&mut self, node: usize, regions: &Regions) -> Vec<PlacedPixmap> {
        let graph = self.graph;
        let layer = &graph.layers[node];
        let mut nested: Vec<Pieces> = graph.nodes.iter().map(|_| Pieces::default()).collect();
        let mut extents = vec![Index::default(); graph.nodes.len()];
        for child in &layer.children {
            let clip = Bounds::covering(child.clip, 1.0);
            let mut pieces = Vec::new();
            for region in regions.iter() {
                let area = region.intersect(clip);
                if !area.is_empty() {
                    pieces.extend(self.stream(child.layer, area));
                }
            }
            nested[child.layer] = Pieces::new(pieces);
            extents[child.layer] = nested[child.layer].index().clone();
        }
        let replay = Replay {
            layer,
            commands: &graph.commands,
            layers: &graph.layers,
            extents: &extents,
            simd_level: graph.simd_level,
        };
        replay
            .windows(regions)
            .iter()
            .filter_map(|window| {
                window.render::<S, F>(self.simd, self.encoded_paints, self.image_resolver, &nested)
            })
            .collect()
    }

    /// Return the output of `node` inside `area`, which lies in its streamed output.
    fn stream(&mut self, node: usize, area: Bounds) -> Vec<PlacedPixmap> {
        let graph = self.graph;
        let graph_node = &graph.nodes[node];
        let plan = &graph_node.streamed;
        let reduced = plan.reduction() > 0;
        let mut output = Vec::new();
        for (region, layout) in plan.parts(area, &graph_node.content, &graph_node.filtered) {
            let source = Pieces::new(self.source(node, &layout.sources(reduced)));
            let Self {
                decimated,
                blurred,
                scratch,
                ..
            } = self;
            let (filtered, blurred) = if reduced {
                let blurred = blurred
                    .entry((node, region))
                    .or_insert_with(|| plan.blur(&layout, &decimated[node], scratch));
                (&decimated[node], Some(&*blurred))
            } else {
                (&source, None)
            };
            output.push(plan.compose(
                &layout,
                &source,
                filtered,
                blurred,
                scratch,
                &mut |pixmap, filter, scratch| F::filter_layer(pixmap, filter, scratch),
            ));
        }
        output
    }
}

/// Return the squares `2^TILE_LEVELS` pixels wide on a grid anchored at the
/// origin that overlap both `area` and `content`, cut to `area`, each once,
/// row by row.
fn content_tiles(area: Bounds, content: &Regions) -> impl Iterator<Item = Bounds> {
    let members: Vec<Bounds> = content
        .touching(area)
        .map(|member| member.intersect(area))
        .filter(|member| !member.is_empty())
        .collect();
    let rows = tile_spans(
        members
            .iter()
            .map(|member| (member.y0, member.y1))
            .collect(),
    );
    rows.flat_map(move |row| {
        let band =
            Bounds::new(Coord::ZERO, row, Coord::from(1), row + Coord::from(1)).refine(TILE_LEVELS);
        let columns = tile_spans(
            members
                .iter()
                .filter(|member| member.y0 < band.y1 && band.y0 < member.y1)
                .map(|member| (member.x0, member.x1))
                .collect(),
        );
        columns.map(move |column| {
            Bounds::new(column, row, column + Coord::from(1), row + Coord::from(1))
                .refine(TILE_LEVELS)
                .intersect(area)
        })
    })
}

/// Return the tiles `2^TILE_LEVELS` pixels wide that the `spans` of pixels
/// along one axis cover, each once, in order.
fn tile_spans(spans: Vec<(Coord, Coord)>) -> impl Iterator<Item = Coord> {
    let mut spans: Vec<(Coord, Coord)> = spans
        .into_iter()
        .map(|(start, end)| {
            let tiles = Bounds::new(start, Coord::ZERO, end, Coord::from(1)).coarsen(TILE_LEVELS);
            (tiles.x0, tiles.x1)
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(Coord, Coord)> = Vec::new();
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged.into_iter().flat_map(|(start, end)| {
        core::iter::successors(Some(start), move |&tile| {
            let next = tile + Coord::from(1);
            (next < end).then_some(next)
        })
    })
}

/// What replaying the windows of one layer needs.
struct Replay<'a> {
    layer: &'a GraphLayer,
    commands: &'a [Command],
    layers: &'a [GraphLayer],
    /// Where the outputs of the filter layers nested in it have content, by layer.
    extents: &'a [Index],
    simd_level: Level,
}

impl Replay<'_> {
    /// Replay the layer into windows covering `regions`.
    fn windows(&self, regions: &Regions) -> Vec<Window> {
        regions
            .iter()
            .flat_map(window_grid)
            .map(|window| self.window(window))
            .collect()
    }

    fn window(&self, window: Bounds) -> Window {
        let origin = window.origin();
        let size = window.size();
        let size = (
            u16::try_from(size.0).unwrap(),
            u16::try_from(size.1).unwrap(),
        );
        let to_window = Affine::translate((-origin.0.to_f64(), -origin.1.to_f64()));
        let dispatcher =
            SingleThreadedDispatcher::window(size.0, size.1, self.simd_level, origin, |target| {
                for clip in &self.layer.clips {
                    clip.push_into(target, to_window);
                }
                for step in &self.layer.steps {
                    match *step {
                        Step::Command(command) => {
                            apply(target, &self.commands[command], to_window);
                        }
                        Step::Nested { layer, push } => {
                            let Command::PushLayer(pushed) = &self.commands[push] else {
                                unreachable!("a nested layer starts with its push");
                            };
                            inject(
                                target,
                                pushed,
                                &self.layers[layer].filter,
                                layer,
                                &self.extents[layer],
                                window,
                                to_window,
                            );
                        }
                    }
                }
            });
        Window { origin, dispatcher }
    }
}

/// A window of a layer's source, replayed into its own dispatcher.
#[derive(Debug)]
struct Window {
    /// The window's origin in root target coordinates.
    origin: Position,
    dispatcher: SingleThreadedDispatcher,
}

impl Window {
    /// Return the bounds of the window content.
    fn content(&self) -> Bounds {
        self.dispatcher.window_content()
    }

    /// Render the window content, or return `None` if it is empty. `nested`
    /// holds the outputs of the nested filter layers, by layer.
    fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        nested: &[Pieces],
    ) -> Option<PlacedPixmap> {
        let encoded_paints = paints_at(encoded_paints, self.origin);
        self.dispatcher
            .render_window::<S, F>(simd, &encoded_paints, image_resolver, nested)
    }
}

/// Return copies of `paints` for sampling at the pixels of a window at `origin`.
fn paints_at(paints: &[EncodedPaint], origin: Position) -> Vec<EncodedPaint> {
    let to_source = Affine::translate((origin.0.to_f64(), origin.1.to_f64()));
    paints
        .iter()
        .map(|paint| {
            let mut paint = paint.clone();
            match &mut paint {
                EncodedPaint::Gradient(gradient) => gradient.transform *= to_source,
                EncodedPaint::Image(image) => image.transform *= to_source,
                EncodedPaint::BlurredRoundedRect(rect) => rect.transform *= to_source,
            }
            paint
        })
        .collect()
}

/// Split `region` into windows on a tile-aligned grid.
fn window_grid(region: Bounds) -> impl Iterator<Item = Bounds> {
    let snapped = region.snap_to_tiles();
    let steps = |start: Coord, end: Coord| {
        core::iter::successors(Some(start), move |&value| {
            let next = value + WINDOW_SIZE.into();
            (next < end).then_some(next)
        })
    };
    steps(snapped.y0, region.y1).flat_map(move |y| {
        steps(snapped.x0, region.x1).map(move |x| {
            Bounds::new(
                x,
                y,
                (x + WINDOW_SIZE.into()).min(region.x1),
                (y + WINDOW_SIZE.into()).min(region.y1),
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
    extents: &Index,
    window: Bounds,
    to_window: Affine,
) {
    let dest = extents
        .touching(window)
        .fold(Bounds::EMPTY, |dest, (_, extent)| {
            dest.union(extent.intersect(window))
        })
        .snap_to_tiles()
        .to_rect_u16(window.origin());
    target.push_rendered_filter_layer(
        layer,
        layer.mask.clone(),
        to_window,
        filter.clone(),
        dest,
        node,
    );
}

fn apply(target: &mut SingleThreadedDispatcher, command: &Command, to_window: Affine) {
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
        } => target.fill_rect_fast(
            &to_window.transform_rect_bbox(*rect),
            paint.clone(),
            *blend_mode,
            mask.clone(),
        ),
        Command::PushLayer(layer) => target.push_layer(
            layer.clip_path.as_ref(),
            layer.fill_rule,
            to_window * layer.clip_transform,
            layer.blend_mode,
            layer.opacity,
            layer.aliasing_threshold,
            layer.mask.clone(),
            None,
        ),
        Command::PopLayer => target.pop_layer(),
        Command::PushClip(clip) => clip.push_into(target, to_window),
        Command::PopClip => target.pop_clip(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use vello_common::color::palette::css::BLUE;
    use vello_common::paint::PremulColor;

    /// Record `filters`, each nested in the one before, around `rects` in root
    /// target coordinates, as `RenderContext` would, and plan them in `view`.
    fn graph(view: Bounds, filters: &[FilterPrimitive], rects: &[Rect]) -> FilterGraph {
        let filter_data = |primitive: &FilterPrimitive| {
            FilterData::new(Filter::from_primitive(primitive.clone()), Affine::IDENTITY)
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
                rect,
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
        /// The pixels of the windows replayed in advance.
        window_pixels: u128,
        strips: usize,
        /// The pixels of the pixmaps the filters run on.
        pixmap_pixels: u128,
        /// The pixels of the decimated sources, and of the tiles of the
        /// full-resolution sources they average.
        decimated_pixels: u128,
        tile_pixels: u128,
    }

    fn view() -> Bounds {
        Bounds::new(0, 0, 64, 64)
    }

    fn work(graph: &FilterGraph) -> Work {
        let windows = || graph.nodes.iter().flat_map(|node| &node.windows);
        let pixmap_pixels = graph
            .nodes
            .iter()
            .flat_map(|node| {
                let kept = Index::new(node.windows.iter().map(Window::content));
                let filtered = if node.plan.reduction() == 0 {
                    kept.clone()
                } else {
                    Index::new(node.decimated.iter().copied())
                };
                node.plan.pixmaps(&kept, &filtered)
            })
            .map(Bounds::area)
            .sum();
        let tiles = |node: &Node| {
            let levels = node.plan.reduction();
            node.decimated
                .iter()
                .flat_map(move |region| {
                    content_tiles(region.refine(levels), &node.decimated_content)
                })
                .map(Bounds::area)
                .sum::<u128>()
        };
        Work {
            nodes: graph.nodes.len(),
            windows: windows().count(),
            window_pixels: windows()
                .map(|window| {
                    let (width, height) = window.dispatcher.size();
                    u128::from(width) * u128::from(height)
                })
                .sum(),
            strips: windows()
                .map(|window| window.dispatcher.strip_count())
                .sum(),
            pixmap_pixels,
            decimated_pixels: graph
                .nodes
                .iter()
                .flat_map(|node| &node.decimated)
                .map(|region| region.area())
                .sum(),
            tile_pixels: graph.nodes.iter().map(tiles).sum(),
        }
    }

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
                view(),
                &vec![shadow.clone(); depth],
                &[Rect::new(x, 16.0, x + 32.0, 48.0)],
            );
            let work = work(&graph);
            // Each layer reads only the copy that its shadows move into the view.
            assert_eq!(work.nodes, depth, "depth {depth}");
            assert_eq!(work.windows, depth, "depth {depth}");
        }
    }

    #[test]
    fn nested_shadows_with_distinct_offsets_read_only_visible_copies() {
        let shadow = |dx: f64, dy: f64| FilterPrimitive::DropShadow {
            dx: dx as f32,
            dy: dy as f32,
            std_deviation: 0.0,
            color: BLUE,
            edge_mode: EdgeMode::None,
        };
        let square = Rect::new(16.0, 16.0, 48.0, 48.0);
        let directions: [fn(f64) -> (f64, f64); 4] = [
            |step| (step, 0.0),
            |step| (-step, step),
            |step| (step, -step),
            |step| (-step, -step),
        ];
        for direction in directions {
            for depth in 1..=16_usize {
                // Every subset of the shadows moves the square to its own place,
                // and only the square itself is visible.
                let shadows: Vec<_> = (0..depth)
                    .map(|level| {
                        let (dx, dy) = direction(256.0 * f64::from(1_u32 << level));
                        shadow(dx, dy)
                    })
                    .collect();
                let work = work(&graph(view(), &shadows, &[square]));
                assert_eq!(work.nodes, depth, "depth {depth}");
                assert!(work.windows <= depth, "depth {depth}: {work:?}");
            }
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
            work(&graph(view(), &[offset, blur.clone()], &[band, square]))
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
            let graph = graph(view, &[blur], &[square]);
            let work = work(&graph);
            assert!(
                work.window_pixels <= 1 << 20
                    && work.pixmap_pixels <= 1 << 20
                    && work.decimated_pixels <= 1 << 20,
                "sigma {std_deviation}: {work:?}"
            );
            // Each pixel of the source within the blur's reach is rendered
            // once, in tiles.
            let levels = 2 * graph.nodes[0].plan.reduction();
            let side = 7.0 * f64::from(std_deviation);
            assert!(
                work.tile_pixels == work.decimated_pixels << levels
                    && work.tile_pixels as f64 <= side * side,
                "sigma {std_deviation}: {work:?}"
            );
        }
    }
}
