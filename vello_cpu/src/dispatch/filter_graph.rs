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
use crate::filter::averaged_image::averaged;
use crate::filter::context::ScratchBuffer;
use crate::filter::placed::{Bounds, Index, Pieces, PlacedPixmap, Regions, pixel_size};
use crate::filter::plan::{FilterPlan, Reach};
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Join, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::clip::PushedClip;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::FilterData;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, ImageSource, Paint};
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
        fold_far_shifts(&self.commands, &mut layers);
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
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    steps: Vec<Step>,
    /// The filter layers nested directly in this one.
    children: Vec<Child>,
    /// A rectangle in root target coordinates that holds everything the
    /// layer's own commands draw, if they draw anything.
    drawn: Option<Rect>,
    /// Offset filters of other layers that move this layer's output, see [`fold_far_shifts`].
    shifts: Vec<FilterData>,
    /// Whether the layer's own offset moves its output, which is not the case
    /// once it is folded into another layer.
    moves: bool,
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
        shifts: Vec::new(),
        moves: true,
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

/// The sum of the offsets of a chain of layers, in root target pixels, from
/// which [`fold_far_shifts`] folds them: below it, the copies the chain moves
/// stay far inside the coordinate domain of [`Bounds`].
const FAR: f64 = (1_u64 << 59) as f64;

/// Fold the offsets of each chain of layers that pass the output of the next
/// one on unchanged, and together move it by at least [`FAR`], into one layer
/// of the chain: into its first layer that does more than move its source and
/// whose own offset reaches that far, else its first layer that does more,
/// else its last layer.
///
/// Every filter moves its output with its source, so the chain's output stays
/// the same, but opposite offsets now cancel before the copy they move is cut
/// off at the edge of the domain. Chains moved less keep their offsets where
/// they are, because a filter's pixmap is anchored where its source has
/// content, and moving that changes the decimation of a blur.
fn fold_far_shifts(commands: &[Command], layers: &mut [GraphLayer]) {
    let next: Vec<Option<usize>> = layers
        .iter()
        .map(|layer| passed_on(commands, layer))
        .collect();
    let mut inner = vec![false; layers.len()];
    for &child in next.iter().flatten() {
        inner[child] = true;
    }
    let offset = |layer: &GraphLayer| {
        let [FilterPrimitive::Offset { .. }] = layer.filter.filter.graph.primitives.as_slice()
        else {
            return None;
        };
        layer
            .filter
            .displacement()
            .map(|displacement| displacement.offset)
    };
    let far = |offset: Option<(f32, f32)>| {
        offset.map_or(0.0, |(dx, dy)| f64::from(dx.abs()) + f64::from(dy.abs()))
    };
    for first in (0..layers.len()).filter(|&layer| !inner[layer]) {
        let mut chain = vec![first];
        while let Some(child) = next[*chain.last().unwrap()] {
            chain.push(child);
        }
        let moved: f64 = chain.iter().map(|&layer| far(offset(&layers[layer]))).sum();
        if moved < FAR {
            continue;
        }
        let filters = || {
            chain
                .iter()
                .copied()
                .filter(|&layer| offset(&layers[layer]).is_none())
        };
        let own = |layer: usize| {
            far(layers[layer]
                .filter
                .displacement()
                .map(|displacement| displacement.offset))
        };
        let target = filters()
            .find(|&layer| own(layer) >= FAR)
            .or_else(|| filters().next())
            .unwrap_or(*chain.last().unwrap());
        for &layer in &chain {
            if layer != target && offset(&layers[layer]).is_some() {
                let filter = layers[layer].filter.clone();
                layers[target].shifts.push(filter);
                layers[layer].moves = false;
            }
        }
    }
}

/// Return the filter layer whose output `layer` takes as its source as it is:
/// its only content, composited without clips, masks, opacity or blending.
fn passed_on(commands: &[Command], layer: &GraphLayer) -> Option<usize> {
    let [Step::Nested { layer: child, push }] = layer.steps.as_slice() else {
        return None;
    };
    let Command::PushLayer(pushed) = &commands[*push] else {
        unreachable!("a nested layer starts with its push");
    };
    let plain = layer.clips.is_empty()
        && pushed.clip_path.is_none()
        && pushed.mask.is_none()
        && pushed.opacity == 1.0
        && pushed.blend_mode == BlendMode::default();
    plain.then_some(*child)
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
    /// Return the index of the encoded paint the command paints with, if any.
    fn paint(&self) -> Option<usize> {
        match self {
            Self::FillPath { paint, .. }
            | Self::StrokePath { paint, .. }
            | Self::FillRect { paint, .. } => match paint {
                Paint::Indexed(paint) => Some(paint.index()),
                Paint::Solid(_) => None,
            },
            Self::PushLayer(_) | Self::PopLayer | Self::PushClip(_) | Self::PopClip => None,
        }
    }

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
    extents: Index,
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

/// A layer whose output is needed in one pixel grid, while it is planned.
struct Need {
    layer: usize,
    level: u32,
    reach: Reach,
    /// A rectangle that holds every region of the output that is needed.
    hull: Bounds,
    /// Where the source the output keeps has content, in the output's grid,
    /// and where the source the filter reads has content, in its grid. The
    /// two are the same when the filter reads its source in the output's grid.
    kept_content: Regions,
    filtered_content: Regions,
    /// Where the output can have content.
    content: Regions,
}

/// Plan the output regions of every layer in every grid it is needed in. The
/// first layer is needed inside `view` in the root grid.
///
/// A region of a source is rendered only where that source has content, so
/// that the copies a drop shadow that keeps its source makes of every copy
/// nested in it are read only where one is visible. Where the layers have
/// content is first found from the children up, inside the regions the
/// parents can read at most, so that it too stays within what is visible.
fn plan(layers: &[GraphLayer], view: Bounds) -> Vec<Planned> {
    // From the parents down: the grids each layer is needed in, and how far.
    let mut hulls: Vec<BTreeMap<u32, Bounds>> = layers.iter().map(|_| BTreeMap::new()).collect();
    hulls[0].insert(0, view);
    let mut needs = Vec::new();
    let mut index = BTreeMap::new();
    for (layer_index, layer) in layers.iter().enumerate() {
        for (level, hull) in core::mem::take(&mut hulls[layer_index]) {
            let reach = Reach::new(&layer.filter, level, &layer.shifts, layer.moves);
            for (level, source) in source_levels(&reach, level, hull) {
                for child in &layer.children {
                    let child_hull = hulls[child.layer].entry(level).or_insert(Bounds::EMPTY);
                    *child_hull = child_hull.union(source);
                }
            }
            index.insert((layer_index, level), needs.len());
            needs.push(Need {
                layer: layer_index,
                level,
                reach,
                hull,
                kept_content: Regions::default(),
                filtered_content: Regions::default(),
                content: Regions::default(),
            });
        }
    }
    // From the children up: where each layer has content.
    for node in (0..needs.len()).rev() {
        let (level, hull) = (needs[node].level, needs[node].hull);
        let layer = &layers[needs[node].layer];
        let reach = &needs[node].reach;
        let (kept_area, filtered_area) = reach.reads(hull);
        let content_in = |level, area: Bounds| source_content(layer, level, area, &needs, &index);
        let (kept, filtered) = if reach.reduction() == 0 {
            let content = content_in(level, kept_area.union(filtered_area));
            (content.clone(), content)
        } else {
            (
                content_in(level, kept_area),
                content_in(level + reach.reduction(), filtered_area),
            )
        };
        let mut content = Regions::default();
        for region in filtered.iter() {
            content.insert(reach.writes(region).intersect(hull));
        }
        if reach.keeps_source() {
            for region in kept.iter() {
                content.insert(region.translate(reach.kept_offset()));
            }
        }
        let need = &mut needs[node];
        need.kept_content = kept;
        need.filtered_content = filtered;
        need.content = content;
    }
    // From the parents down: the regions of each layer that are needed.
    let mut outputs: Vec<Regions> = needs.iter().map(|_| Regions::default()).collect();
    outputs[0].insert(view);
    let mut planned = Vec::with_capacity(needs.len());
    for (node, need) in needs.iter().enumerate() {
        let plan = FilterPlan::new(need.reach.clone(), &core::mem::take(&mut outputs[node]));
        let (kept, filtered) = plan.source_regions();
        let reduction = plan.reduction();
        let (fine, coarse) = if reduction == 0 {
            let mut fine = kept;
            filtered.iter().for_each(|region| fine.insert(region));
            (fine, Regions::default())
        } else {
            (kept, filtered)
        };
        let layer = &layers[need.layer];
        for (level, sources) in [(need.level, &fine), (need.level + reduction, &coarse)] {
            for child in &layer.children {
                if let Some(&child_node) = index.get(&(child.layer, level)) {
                    let clip = Bounds::covering(child.clip, 1.0 / pixel_size(level));
                    for content in needs[child_node].content.iter() {
                        let content = content.intersect(clip);
                        for source in sources.touching(content) {
                            outputs[child_node].insert(source.intersect(content));
                        }
                    }
                }
            }
        }
        planned.push(Planned {
            layer: need.layer,
            level: need.level,
            plan,
            regions: intersection(&fine, &need.kept_content),
            coarse: intersection(&coarse, &need.filtered_content),
        });
    }
    planned
}

/// Return the grids in which a layer that reads and writes as `reach` reads
/// its source when its output is needed inside `hull` in the grid of `level`,
/// with the bounds it reads there: the kept source in the output's grid and
/// the source the filter reads in its own, one grid if they coincide.
fn source_levels(reach: &Reach, level: u32, hull: Bounds) -> impl Iterator<Item = (u32, Bounds)> {
    let (kept, filtered) = reach.reads(hull);
    let reduction = reach.reduction();
    let levels = if reduction == 0 {
        [(level, kept.union(filtered)), (level, Bounds::EMPTY)]
    } else {
        [(level, kept), (level + reduction, filtered)]
    };
    levels.into_iter().filter(|(_, area)| !area.is_empty())
}

/// Return where the source of `layer` has content inside `area` in the grid of
/// `level`: its own drawing and the outputs of the nodes of its children in
/// that grid, inside their clips.
fn source_content(
    layer: &GraphLayer,
    level: u32,
    area: Bounds,
    needs: &[Need],
    index: &BTreeMap<(usize, u32), usize>,
) -> Regions {
    let scale = 1.0 / pixel_size(level);
    let mut content = Regions::default();
    if let Some(drawn) = layer.drawn {
        content.insert(Bounds::covering(drawn, scale).intersect(area));
    }
    for child in &layer.children {
        let Some(&node) = index.get(&(child.layer, level)) else {
            continue;
        };
        let clip = Bounds::covering(child.clip, scale).intersect(area);
        for region in needs[node].content.iter() {
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
            let contents = |windows: &[Window]| Index::new(windows.iter().map(Window::content));
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
                extents: Index::new(extents),
            });
        }
        Self {
            nodes: nodes.into_iter().map(Option::unwrap).collect(),
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
            .to_rect_u16((0, 0))
    }

    /// Render the layer and the filters nested in it, and return its output in the root grid.
    pub(crate) fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> Pieces {
        let mut outputs: Vec<Pieces> = self.nodes.iter().map(|_| Pieces::default()).collect();
        let mut scratch = ScratchBuffer::new();
        for (index, node) in self.nodes.iter().enumerate().rev() {
            let render = |windows: &[Window]| {
                Pieces::new(
                    windows
                        .iter()
                        .filter_map(|window| {
                            window.render::<S, F>(simd, encoded_paints, image_resolver, &outputs)
                        })
                        .collect(),
                )
            };
            let kept = render(&node.windows);
            let filtered = if node.plan.reduction() == 0 {
                None
            } else {
                Some(render(&node.coarse))
            };
            outputs[index] = Pieces::new(node.plan.render(
                &kept,
                filtered.as_ref().unwrap_or(&kept),
                &mut scratch,
                &mut |pixmap, filter, scratch| F::filter_layer(pixmap, filter, scratch),
            ));
        }
        core::mem::take(&mut outputs[0])
    }
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
        let size = (
            u16::try_from(window.width()).unwrap(),
            u16::try_from(window.height()).unwrap(),
        );
        let to_window = Affine::translate((-origin.0 as f64, -origin.1 as f64))
            * Affine::scale(1.0 / pixel_size(level));
        let dispatcher =
            SingleThreadedDispatcher::window(size.0, size.1, self.simd_level, origin, |target| {
                for clip in &self.layer.clips {
                    clip.push_into(target, to_window);
                }
                for step in &self.layer.steps {
                    match *step {
                        Step::Command(command) => {
                            apply(target, &self.commands[command], to_window, masks);
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
            });
        let mut paints: Vec<usize> = self
            .layer
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Command(command) => self.commands[*command].paint(),
                Step::Nested { .. } => None,
            })
            .collect();
        paints.sort_unstable();
        paints.dedup();
        Window {
            origin,
            size,
            level,
            paints,
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
    let coarse = |index: u16| usize::from(index).checked_shr(levels).unwrap_or(0);
    let length = |length: u16| {
        if length == 0 {
            0
        } else {
            coarse(length - 1) + 1
        }
    };
    let (width, height) = (length(mask.width()), length(mask.height()));
    let mut sums = vec![0_u64; width * height];
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            sums[coarse(y) * width + coarse(x)] += u64::from(mask.sample(x, y));
        }
    }
    // Divide by the `4^levels` pixels of each square, rounding to nearest.
    let average = |sum: u64| match levels {
        0 => sum,
        _ => (sum.checked_shr(2 * levels - 1).unwrap_or(0) + 1) >> 1,
    };
    let data = sums.into_iter().map(|sum| average(sum) as u8).collect();
    Mask::from_parts(data, width as u16, height as u16)
}

/// A window of a layer's source, replayed into its own dispatcher.
#[derive(Debug)]
struct Window {
    /// The window's origin, in the grid of root target pixels `2^level` wide.
    origin: (i64, i64),
    size: (u16, u16),
    level: u32,
    /// The indices of the encoded paints the window's commands paint with.
    paints: Vec<usize>,
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
        nested: &[Pieces],
    ) -> Option<PlacedPixmap> {
        let mut encoded_paints = paints_at(encoded_paints, self.to_source());
        // In the root grid, images are sampled like without filters.
        if self.level > 0 {
            for &index in &self.paints {
                let EncodedPaint::Image(image) = &encoded_paints[index] else {
                    continue;
                };
                let pixmap = match &image.source {
                    ImageSource::Pixmap(pixmap) => Some(pixmap.clone()),
                    ImageSource::OpaqueId { id, .. } => image_resolver.resolve(*id),
                    ImageSource::ExternalTexture { .. } => None,
                };
                if let Some(averaged) =
                    pixmap.and_then(|pixmap| averaged(image, &pixmap, self.size.0, self.size.1))
                {
                    encoded_paints[index] = EncodedPaint::Image(averaged);
                }
            }
        }
        self.dispatcher
            .render_window::<S, F>(simd, &encoded_paints, image_resolver, nested)
    }

    /// Return the transform from the window's pixels to root target coordinates.
    fn to_source(&self) -> Affine {
        Affine::scale(pixel_size(self.level))
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
    extents: &Index,
    window: Bounds,
    to_window: Affine,
    masks: &mut Masks,
) {
    let dest = extents
        .touching(window)
        .fold(Bounds::EMPTY, |dest, (_, extent)| {
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
        /// The pixels of the windows the source is rasterized in.
        window_pixels: u128,
        strips: usize,
        /// The pixels of the pixmaps the filters run on.
        pixmap_pixels: u128,
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
                let kept = Index::new(node.windows.iter().map(Window::content));
                let filtered = if node.plan.reduction() == 0 {
                    kept.clone()
                } else {
                    Index::new(node.coarse.iter().map(Window::content))
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
                    u128::from(width) * u128::from(height)
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
                let work = work(&graph(VIEW, &shadows, &[square]));
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
