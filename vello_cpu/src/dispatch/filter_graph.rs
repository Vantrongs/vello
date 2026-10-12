// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter layers, rendered from the parts of their source that their visible
//! output depends on.
//!
//! A filter layer records its commands instead of drawing them. When it is
//! popped, it and the filter layers nested in it are placed as the recorder
//! places filter layers, see [`Node`]: each runs its filter on a pixmap
//! around the content of its source, and its output is composited into its
//! parent's source. Those rectangles lie in coordinates of the root target
//! that are exact integers of any size, see [`Coord`], so they can be far
//! larger than any pixmap.
//!
//! The output is then computed only where the view shows it: each filter
//! computes the pixels of an area from the pixels of its source that they
//! depend on, see [`Pyramid`] for blurs, and the source is rendered for those
//! pixels in windows of at most 65532 pixels a side, the most that strips,
//! tiles and fine rasterization address, with the output of each nested filter
//! computed the same way where a window shows it. Pixels are those of the
//! filters run on whole pixmaps, while memory follows the view and the work
//! follows the source the visible output depends on.

use super::Dispatcher;
use super::single_threaded::SingleThreadedDispatcher;
use crate::RasterizerSettings;
use crate::filter::colorize_shadow;
use crate::filter::context::ScratchBuffer;
use crate::filter::pixmap::FilterPixmap;
use crate::filter::placed::{Bounds, Coord, Padding, PlacedPixmap, Position, Regions};
use crate::filter::plan::{Effect, Reach};
use crate::filter::pyramid::{Blur, Pyramid};
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Join, Rect, Stroke};
use crate::peniko::{BlendMode, Compose, Fill};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::marker::PhantomData;
use vello_common::clip::PushedClip;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;

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
    /// Whether the area the output is composited into is visible, see [`shows_area`].
    area_shown: bool,
    commands: Vec<Command>,
    /// The number of open layers, the recorded layer included.
    depth: usize,
}

impl Recording {
    /// Start recording a filter layer whose output is visible in `view`.
    pub(crate) fn new(
        filter: FilterData,
        view: Bounds,
        clips: Vec<Clip>,
        area_shown: bool,
    ) -> Self {
        Self {
            filter,
            view,
            clips,
            area_shown,
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
        FilterGraph::new(self.commands, layers, self.view, self.area_shown, level)
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
}

impl GraphLayer {
    /// Return for each of the layer's own commands that draws a rectangle in
    /// root target coordinates that holds what it draws.
    fn drawn(&self) -> impl Iterator<Item = Rect> + '_ {
        self.steps.iter().filter_map(|step| match step {
            Step::Command { drawn, .. } => *drawn,
            Step::Nested { .. } => None,
        })
    }
}

/// A filter layer nested directly in another.
#[derive(Clone, Copy, Debug)]
struct Child {
    layer: usize,
    /// A rectangle in root target coordinates outside of which the clip
    /// paths of the layers it is composited through hide it.
    clip: Rect,
    /// A rectangle in root target coordinates outside of which the clip paths
    /// of the layers it is nested in cut it from the content bounds of its
    /// parent, [`UNBOUNDED`] if none do.
    enclosing: Rect,
    /// Whether the area it is composited into is visible in its parent, see [`shows_area`].
    area_shown: bool,
}

/// A recorded command, or a nested filter layer rendered on its own.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// The command `command`, which draws inside `drawn` if it draws.
    Command { command: usize, drawn: Option<Rect> },
    /// The filter layer `layer`, pushed by the command `push`.
    Nested { layer: usize, push: usize },
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
    });
    let mut active_clips = clips.clone();
    let mut clip_bounds = ClipBounds::new(&clips);
    let mut steps = Vec::new();
    let mut children = Vec::new();
    let mut command = range.start;
    while command < range.end {
        if let Command::PushLayer(layer) = &commands[command]
            && let Some(filter) = &layer.filter
        {
            let end = matching_pop(commands, command);
            let enclosing = clip_bounds.composited();
            let child = parse(
                commands,
                command + 1..end,
                filter.clone(),
                active_clips.clone(),
                layers,
            );
            children.push(Child {
                layer: child,
                clip: layer.clip_bounds().intersect(enclosing),
                enclosing,
                area_shown: clip_bounds.area_shown() || shows_area(layer.blend_mode),
            });
            steps.push(Step::Nested {
                layer: child,
                push: command,
            });
            // Clips pushed or popped inside the filter layer stay in effect after it.
            for (inner, nested) in commands.iter().enumerate().take(end).skip(command + 1) {
                if track_clips(nested, &mut active_clips) {
                    clip_bounds.track(nested);
                    steps.push(Step::Command {
                        command: inner,
                        drawn: None,
                    });
                }
            }
            command = end + 1;
            continue;
        }
        track_clips(&commands[command], &mut active_clips);
        clip_bounds.track(&commands[command]);
        let drawn = commands[command]
            .bounds()
            .map(|bounds| bounds.intersect(clip_bounds.drawn()));
        steps.push(Step::Command { command, drawn });
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
    /// For each layer, whether an open layer shows the area it is composited into.
    shown: Vec<bool>,
}

impl ClipBounds {
    /// Start with `clips` in effect, which can be popped like the others.
    fn new(clips: &[Clip]) -> Self {
        let mut bounds = Self {
            clips: vec![UNBOUNDED],
            layers: vec![UNBOUNDED],
            shown: vec![false],
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

    /// Return whether an open layer shows the area it is composited into.
    fn area_shown(&self) -> bool {
        *self.shown.last().unwrap()
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
                self.shown
                    .push(self.area_shown() || shows_area(layer.blend_mode));
            }
            Command::PopLayer if self.layers.len() > 1 => {
                self.layers.pop();
                self.shown.pop();
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

/// Return whether compositing a transparent pixel with `blend_mode` changes
/// what lies under it, which makes the area a layer is composited into visible.
pub(crate) fn shows_area(blend_mode: BlendMode) -> bool {
    matches!(
        blend_mode.compose,
        Compose::Clear
            | Compose::Copy
            | Compose::SrcIn
            | Compose::DestIn
            | Compose::SrcOut
            | Compose::DestAtop
    )
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

/// A recorded filter layer and the filter layers nested in it, placed.
#[derive(Debug)]
pub(crate) struct FilterGraph {
    commands: Vec<Command>,
    layers: Vec<GraphLayer>,
    /// One node for each layer, children after their parents. The first is
    /// the recorded layer.
    nodes: Vec<Node>,
    /// Where the output of the recorded layer is visible.
    view: Bounds,
    simd_level: Level,
}

/// A filter layer placed as the recorder places filter layers in a target
/// large enough to hold all of its content, in root target coordinates.
///
/// The filter runs on a pixmap that covers the bounds of the content the
/// layer's source draws and of the pixmaps of the nested layers, widened by
/// the filter padding, and its output is composited into its parent where
/// that pixmap lies. The bounds of the content are those of the strips the
/// source draws, so they are found by replaying it in windows, see
/// [`FilterGraph::content_bounds`]. A command that draws without bounds, such
/// as a rectangle with an infinite side, is bounded by the viewport the
/// recorder draws it in: the view widened by the source paddings of the
/// layers down to this one.
///
/// Where the content ends matters only to a filter that reads past it, a
/// blur or a flood, to a layer composited into an area that is visible, see
/// [`shows_area`], and to the layers these are nested in. The other pixmaps
/// cover the bounds of what the commands can draw instead, which leaves the
/// output as it is and saves replaying content that is not visible.
#[derive(Debug)]
struct Node {
    reach: Reach,
    /// The pixmap the filter runs on, where the output is composited into the
    /// parent; empty if the source has no content.
    pixmap: Bounds,
    /// Where the source can have content inside the pixmap.
    source: Regions,
    /// Where the source of the blur can have content, relative to the pixmap.
    blurred: Regions,
    /// Where the output of the filter can have content, before a drop shadow
    /// puts the source over it.
    filtered: Regions,
    /// Where the output can have content.
    content: Regions,
}

impl FilterGraph {
    /// Place the layers, the first of which is composited into an area that
    /// is visible if `area_shown` is set.
    fn new(
        commands: Vec<Command>,
        layers: Vec<GraphLayer>,
        view: Bounds,
        area_shown: bool,
        simd_level: Level,
    ) -> Self {
        let count = layers.len();
        let reaches: Vec<Reach> = layers
            .iter()
            .map(|layer| Reach::new(&layer.filter))
            .collect();
        // From the parents down: the viewports that bound unbounded commands,
        // and whether the output depends on where the content ends, see `Node`.
        let mut viewports = vec![Bounds::EMPTY; count];
        let mut exact = vec![false; count];
        viewports[0] = view.expand(&reaches[0].source_padding);
        exact[0] = area_shown || reaches[0].reads_extent();
        for (index, layer) in layers.iter().enumerate() {
            for child in &layer.children {
                viewports[child.layer] =
                    viewports[index].expand(&reaches[child.layer].source_padding);
                exact[child.layer] =
                    exact[index] || child.area_shown || reaches[child.layer].reads_extent();
            }
        }
        let mut graph = Self {
            commands,
            layers,
            nodes: Vec::with_capacity(count),
            view,
            simd_level,
        };
        // From the children up: a layer's pixmap covers its children's.
        let mut nodes: Vec<Option<Node>> = (0..count).map(|_| None).collect();
        for (index, reach) in reaches.into_iter().enumerate().rev() {
            let node = graph.place(index, reach, &viewports[index], exact[index], &nodes);
            nodes[index] = Some(node);
        }
        graph.nodes = nodes.into_iter().map(Option::unwrap).collect();
        graph
    }

    /// Place layer `index`, whose children are placed in `nodes`; `viewport`
    /// bounds its unbounded commands. Unless `exact` is set, the pixmap covers
    /// the bounds of what the source can draw rather than of what it draws.
    fn place(
        &self,
        index: usize,
        reach: Reach,
        viewport: &Bounds,
        exact: bool,
        nodes: &[Option<Node>],
    ) -> Node {
        let layer = &self.layers[index];
        let child = |child: &Child| nodes[child.layer].as_ref().unwrap();
        let drawn: Vec<Option<Bounds>> = layer.drawn().map(Bounds::covering).collect();
        // Pixmaps outside the clip paths of regular layers count in full, see
        // `CommandRecorder::pop_layer`.
        let mut bounds = Bounds::EMPTY;
        let mut candidates: Vec<Bounds> = drawn
            .iter()
            .map(|drawn| drawn.clone().unwrap_or_else(|| viewport.clone()))
            .collect();
        for nested in &layer.children {
            let pixmap = &child(nested).pixmap;
            match Bounds::covering(nested.enclosing) {
                Some(enclosing) => candidates.push(enclosing.intersect(pixmap)),
                None => bounds = bounds.union(pixmap),
            }
        }
        let pixmap_of = |node: usize| nodes[node].as_ref().unwrap().pixmap.clone();
        let bounds = if exact {
            self.content_bounds(index, candidates, bounds, &pixmap_of)
        } else {
            candidates
                .iter()
                .fold(bounds, |bounds, candidate| bounds.union(candidate))
        };
        let pixmap = bounds.expand(&reach.filter_padding).snap_to_tiles();
        let mut source = Regions::default();
        for drawn in drawn {
            source.insert(drawn.map_or(pixmap.clone(), |drawn| drawn.intersect(&pixmap)));
        }
        for nested in &layer.children {
            let node = child(nested);
            let shown = node.pixmap.clip(nested.clip).intersect(&pixmap);
            for content in node.content.touching(&shown) {
                source.insert(content.intersect(&shown));
            }
        }
        let local = |regions: &Regions| {
            let origin = pixmap.origin();
            let back = (-&origin.0, -&origin.1);
            let mut local = Regions::default();
            for region in regions.iter() {
                local.insert(region.translate(&back));
            }
            local
        };
        let moved = |offset: &Position| {
            let mut moved = Regions::default();
            for region in source.iter() {
                moved.insert(region.translate(offset).intersect(&pixmap));
            }
            moved
        };
        let (blurred, filtered) = match &reach.effect {
            Effect::Flood => (Regions::default(), Regions::of(pixmap.clone())),
            Effect::Offset(offset) => (Regions::default(), moved(offset)),
            Effect::Blur(None) => (Regions::default(), source.clone()),
            Effect::Blur(Some(blur)) => (local(&source), spread(&source, blur, &pixmap)),
            Effect::Shadow { offset, blur, .. } => {
                let moved = moved(offset);
                match blur {
                    Some(blur) => (local(&moved), spread(&moved, blur, &pixmap)),
                    None => (Regions::default(), moved),
                }
            }
        };
        let mut content = filtered.clone();
        if let Effect::Shadow {
            keeps_source: true, ..
        } = &reach.effect
        {
            for region in source.iter() {
                content.insert(region.clone());
            }
        }
        Node {
            reach,
            pixmap,
            source,
            blurred,
            filtered,
            content,
        }
    }

    /// Return the bounds of the content of the source of layer `index`, given
    /// `bounds` that it covers: the union of the content bounds of windows
    /// over the cells of a grid that hold `candidates`, which hold everything
    /// it draws that can widen `bounds`. `pixmap` returns the pixmap of each
    /// nested layer.
    ///
    /// Cells inside the bounds found so far cannot widen them, so the
    /// candidates that reach furthest are replayed first, and a candidate
    /// spanning several cells is replayed at its corners before the halves of
    /// what remains of it: content that fills a candidate is bounded by its
    /// corners, so the cells replayed follow the extent of the content rather
    /// than its area.
    fn content_bounds(
        &self,
        index: usize,
        mut candidates: Vec<Bounds>,
        mut bounds: Bounds,
        pixmap: &dyn Fn(usize) -> Bounds,
    ) -> Bounds {
        candidates.retain(|candidate| !candidate.is_empty());
        let extent = candidates
            .iter()
            .fold(Bounds::EMPTY, |extent, candidate| extent.union(candidate));
        if extent.is_empty() {
            return bounds;
        }
        let replay = |cell: &(Coord, Coord), bounds: &mut Bounds| {
            let area = cell_bounds(cell).intersect(&extent).snap_to_tiles();
            if !bounds.contains(&area) {
                let window = self.window(index, &area, pixmap);
                *bounds = bounds.union(&window.window_content());
            }
        };
        // The stack pops the candidates reaching furthest first.
        let reaching = |side: fn(&Bounds) -> &Coord, last: bool| {
            let mut ordered: Vec<usize> = (0..candidates.len()).collect();
            ordered.sort_by(|&a, &b| side(&candidates[a]).cmp(side(&candidates[b])));
            if last {
                ordered.last().copied()
            } else {
                ordered.first().copied()
            }
        };
        let mut extremes: Vec<usize> = [
            reaching(|bounds| &bounds.x0, false),
            reaching(|bounds| &bounds.y0, false),
            reaching(|bounds| &bounds.x1, true),
            reaching(|bounds| &bounds.y1, true),
        ]
        .into_iter()
        .flatten()
        .collect();
        extremes.sort_unstable();
        extremes.dedup();
        // Each area with the corners already replayed: top left, top right,
        // bottom left and bottom right.
        let mut pending: Vec<(Bounds, [bool; 4])> = Vec::with_capacity(candidates.len());
        for (index, candidate) in candidates.iter().enumerate() {
            if extremes.binary_search(&index).is_err() {
                pending.push((candidate.clone(), [false; 4]));
            }
        }
        pending.extend(
            extremes
                .iter()
                .map(|&index| (candidates[index].clone(), [false; 4])),
        );
        while let Some((area, replayed)) = pending.pop() {
            if bounds.contains(&area) {
                continue;
            }
            let first = cell_of(&area.x0, &area.y0);
            let last = cell_of(&(&area.x1 - 1), &(&area.y1 - 1));
            let corners = [
                (first.0.clone(), first.1.clone()),
                (last.0.clone(), first.1.clone()),
                (first.0.clone(), last.1.clone()),
                (last.0.clone(), last.1.clone()),
            ];
            for (corner, cell) in corners.iter().enumerate() {
                if !replayed[corner] && !corners[..corner].contains(cell) {
                    replay(cell, &mut bounds);
                }
            }
            if first == last || bounds.contains(&area) {
                continue;
            }
            // Split at a cell edge across the axis with more cells; each half
            // keeps two of the corners.
            let (mut low, mut high) = (area.clone(), area.clone());
            let middle = |first: &Coord, last: &Coord| {
                ((first + last).div_floor(&Coord::from(2_i64)) + 1).mul(&Coord::from(PROBE_SIZE))
            };
            let kept = if &last.0 - &first.0 >= &last.1 - &first.1 {
                low.x1 = middle(&first.0, &last.0);
                high.x0 = low.x1.clone();
                [[true, false, true, false], [false, true, false, true]]
            } else {
                low.y1 = middle(&first.1, &last.1);
                high.y0 = low.y1.clone();
                [[true, true, false, false], [false, false, true, true]]
            };
            pending.push((low, kept[0]));
            pending.push((high, kept[1]));
        }
        bounds
    }

    /// Replay layer `index` into a window covering `area`, which starts on a
    /// tile corner and spans at most [`WINDOW_SIZE`] pixels on each side, with
    /// nested layers composited where `pixmap` says.
    fn window(
        &self,
        index: usize,
        area: &Bounds,
        pixmap: &dyn Fn(usize) -> Bounds,
    ) -> SingleThreadedDispatcher {
        let layer = &self.layers[index];
        let origin = area.origin();
        let side = |length: Coord| u16::try_from(length.to_i64().unwrap()).unwrap();
        let size = (side(area.width()), side(area.height()));
        let to_window = Affine::translate((-origin.0.to_f64(), -origin.1.to_f64()));
        SingleThreadedDispatcher::window(size, self.simd_level, origin.clone(), |target| {
            for clip in &layer.clips {
                clip.push_into(target, to_window);
            }
            for step in &layer.steps {
                match *step {
                    Step::Command { command, drawn } => {
                        // A command drawing outside the window has no strips in it.
                        let outside = drawn
                            .and_then(Bounds::covering)
                            .is_some_and(|drawn| !drawn.touches(area));
                        if !outside {
                            apply(target, &self.commands[command], to_window);
                        }
                    }
                    Step::Nested { layer, push } => {
                        let Command::PushLayer(pushed) = &self.commands[push] else {
                            unreachable!("a nested layer starts with its push");
                        };
                        target.push_rendered_filter_layer(
                            pushed,
                            pushed.mask.clone(),
                            to_window,
                            self.layers[layer].filter.clone(),
                            pixmap(layer).intersect(area).to_rect_u16(&origin),
                            layer,
                        );
                    }
                }
            }
        })
    }

    /// Return where the output of the recorded layer is composited in the view.
    fn placement(&self) -> Bounds {
        self.nodes[0].pixmap.intersect(&self.view)
    }

    /// Return where the output of the recorded layer is composited in the view.
    pub(crate) fn placement_in_view(&self) -> RectU16 {
        self.placement().to_rect_u16(&(Coord::ZERO, Coord::ZERO))
    }

    /// Render the visible output of the recorded layer, placed in the root grid.
    pub(crate) fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> PlacedPixmap {
        let rendering = Rendering::<S, F> {
            graph: self,
            simd,
            encoded_paints,
            image_resolver,
            pyramids: self.nodes.iter().map(|_| RefCell::new(None)).collect(),
            scratch: RefCell::new(ScratchBuffer::new()),
            kernel: PhantomData,
        };
        let area = self.placement();
        if area.is_empty() {
            return PlacedPixmap::new(area);
        }
        PlacedPixmap {
            pixmap: rendering.output(0, &area),
            bounds: area,
        }
    }
}

/// Return where the output of a blur of `content` on `pixmap` can have content.
fn spread(content: &Regions, blur: &Blur, pixmap: &Bounds) -> Regions {
    let mut support = blur.support();
    let mut shifts = vec![(Coord::ZERO, Coord::ZERO)];
    if blur.edge_mode == EdgeMode::Wrap {
        // A level wraps at its size, whose pixels each cover `2^level` pixels
        // of the pixmap, so the copies that wrapping brings in lie one pixmap
        // and less than `2^decimations` pixels away.
        support = support + Coord::from(2_i64).shl(blur.decimations);
        let (width, height) = (pixmap.width(), pixmap.height());
        shifts = [-&width, Coord::ZERO, width]
            .into_iter()
            .flat_map(|x| [-&height, Coord::ZERO, height.clone()].map(|y| (x.clone(), y)))
            .collect();
    }
    let padding = Padding::uniform(support);
    let mut spread = Regions::default();
    for region in content.iter() {
        for shift in &shifts {
            spread.insert(region.translate(shift).expand(&padding).intersect(pixmap));
        }
    }
    spread
}

/// The width and height of the cells replayed to find the bounds of content,
/// small so that finding the bounds of a large area of content takes little memory.
const PROBE_SIZE: i64 = 256;

/// Return the cell replayed to find the bounds of content that holds `(x, y)`.
fn cell_of(x: &Coord, y: &Coord) -> (Coord, Coord) {
    let size = Coord::from(PROBE_SIZE);
    (x.div_floor(&size), y.div_floor(&size))
}

/// Return the bounds of the cell `(column, row)` replayed to find the bounds of content.
fn cell_bounds((column, row): &(Coord, Coord)) -> Bounds {
    let size = Coord::from(PROBE_SIZE);
    let (x0, y0) = (column.mul(&size), row.mul(&size));
    Bounds::new(x0.clone(), y0.clone(), &x0 + &size, &y0 + &size)
}

/// The state of rendering a filter graph.
struct Rendering<'a, S: Simd, F> {
    graph: &'a FilterGraph,
    simd: S,
    encoded_paints: &'a [EncodedPaint],
    image_resolver: &'a dyn ImageResolver,
    /// The blurs of the layers that blur, prepared when first needed.
    pyramids: Vec<RefCell<Option<Pyramid>>>,
    scratch: RefCell<ScratchBuffer>,
    kernel: PhantomData<F>,
}

impl<S: Simd, F: FineKernel<S>> Rendering<'_, S, F> {
    /// Return the output of layer `index` inside `area`, which lies in its pixmap.
    fn output(&self, index: usize, area: &Bounds) -> FilterPixmap {
        let node = &self.graph.nodes[index];
        if !node.content.overlaps(area) {
            let (width, height) = area.size();
            return FilterPixmap::new(width, height);
        }
        match &node.reach.effect {
            Effect::Flood => {
                let (width, height) = area.size();
                let mut pixmap = FilterPixmap::new(width, height);
                let filter = &self.graph.layers[index].filter;
                F::filter_layer(
                    &mut pixmap,
                    PreparedFilter::new(&filter.filter, &filter.transform),
                    &mut self.scratch.borrow_mut(),
                );
                pixmap
            }
            Effect::Offset(offset) => self.moved(index, area, offset),
            Effect::Blur(None) => self.source(index, area).pixmap,
            Effect::Blur(Some(blur)) => self.blurred(index, area, blur, None),
            Effect::Shadow {
                offset,
                blur,
                color,
                keeps_source,
            } => {
                let mut shadow = match blur {
                    _ if !node.filtered.overlaps(area) => {
                        let (width, height) = area.size();
                        FilterPixmap::new(width, height)
                    }
                    Some(blur) => self.blurred(index, area, blur, Some(offset)),
                    None => self.moved(index, area, offset),
                };
                let source = keeps_source.then(|| self.source(index, area).pixmap);
                colorize_shadow(&mut shadow, *color, source.as_ref());
                shadow
            }
        }
    }

    /// Return the source of layer `index` moved by `offset` inside `area`,
    /// which lies in its pixmap; what the move takes out of the pixmap is lost.
    fn moved(&self, index: usize, area: &Bounds, offset: &Position) -> FilterPixmap {
        let node = &self.graph.nodes[index];
        let from = area
            .translate(&(-&offset.0, -&offset.1))
            .intersect(&node.pixmap);
        let mut moved = PlacedPixmap::new(area.clone());
        if !from.is_empty() {
            moved.copy_from(&self.source(index, &from).translate(offset), area);
        }
        moved.pixmap
    }

    /// Return the blur of the source of layer `index`, moved by `offset` if
    /// there is one, inside `area`, which lies in its pixmap.
    fn blurred(
        &self,
        index: usize,
        area: &Bounds,
        blur: &Blur,
        offset: Option<&Position>,
    ) -> FilterPixmap {
        let node = &self.graph.nodes[index];
        let origin = node.pixmap.origin();
        let mut pyramid = self.pyramids[index].borrow_mut();
        let pyramid = pyramid.get_or_insert_with(|| {
            Pyramid::new(
                blur.clone(),
                node.pixmap.width(),
                node.pixmap.height(),
                &node.blurred,
            )
        });
        let back = (-&origin.0, -&origin.1);
        pyramid.blur(&area.translate(&back), &mut |local| {
            let area = local.translate(&origin);
            match offset {
                Some(offset) => self.moved(index, &area, offset),
                None => self.source(index, &area).pixmap,
            }
        })
    }

    /// Render the source of layer `index` inside `area`, which lies in its pixmap.
    fn source(&self, index: usize, area: &Bounds) -> PlacedPixmap {
        let graph = self.graph;
        let node = &graph.nodes[index];
        let mut source = PlacedPixmap::new(area.clone());
        // Windows are rendered in whole tiles.
        let content = node.source.hull_within(area).snap_to_tiles();
        for shown in split(&content, WINDOW_SIZE) {
            let mut nested = BTreeMap::new();
            for child in &graph.layers[index].children {
                let child_node = &graph.nodes[child.layer];
                let needed = child_node.pixmap.intersect(&shown);
                if child_node.content.overlaps(&needed) {
                    nested.insert(
                        child.layer,
                        PlacedPixmap {
                            pixmap: self.output(child.layer, &needed),
                            bounds: needed,
                        },
                    );
                }
            }
            let window = graph.window(index, &shown, &|layer| graph.nodes[layer].pixmap.clone());
            let encoded_paints = paints_at(self.encoded_paints, &shown.origin());
            if let Some(rendered) = window.render_window::<S, F>(
                self.simd,
                &encoded_paints,
                self.image_resolver,
                &nested,
                &shown,
            ) {
                source.copy_from(&rendered, area);
            }
        }
        source
    }
}

/// Width and height of a window, the largest tile multiple in the `u16` domain.
const WINDOW_SIZE: i64 = 65532;

/// Return `bounds` split into pieces at most `size` wide and high, from its top left.
fn split(bounds: &Bounds, size: i64) -> Vec<Bounds> {
    let steps = |start: &Coord, end: &Coord| {
        let count = (end - start).div_floor(&Coord::from(size)).to_usize() + 1;
        (0..count)
            .map(|step| start + i64::try_from(step).unwrap() * size)
            .filter(|from| from < end)
            .map(|from| {
                let to = (&from + size).min(end.clone());
                (from, to)
            })
            .collect::<Vec<_>>()
    };
    if bounds.is_empty() {
        return Vec::new();
    }
    let columns = steps(&bounds.x0, &bounds.x1);
    steps(&bounds.y0, &bounds.y1)
        .into_iter()
        .flat_map(|(y0, y1)| {
            columns
                .iter()
                .map(move |(x0, x1)| Bounds::new(x0.clone(), y0.clone(), x1.clone(), y1.clone()))
        })
        .collect()
}

/// Return copies of `paints` for sampling at the pixels of a window at `origin`.
fn paints_at(paints: &[EncodedPaint], origin: &Position) -> Vec<EncodedPaint> {
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

#[cfg(all(test, feature = "std"))]
mod tests;
