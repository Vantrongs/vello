// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter layers whose source does not fit the `u16` coordinate domain.
//!
//! Strips, tiles and fine rasterization address pixels with `u16`. A filter can
//! still need source pixels beyond that domain: its halo reaches past a target
//! that is 65535 pixels wide, or an offset moves geometry from far away into
//! view. Such a layer records its commands instead of drawing them. When it is
//! popped, they are replayed into windows of at most 65532 pixels that cover the
//! source region the visible output depends on. At rasterization the windows are
//! stitched into one `u32`-sized pixmap, which is filtered as a whole, and the
//! part inside the parent viewport is composited like any other filter layer.
//!
//! Filters nested inside are rendered the same way and handed to the windows as
//! finished pixmaps, so every filter sees its whole source at once and the
//! result does not depend on where the window edges fall.

use super::Dispatcher;
use super::single_threaded::SingleThreadedDispatcher;
use crate::filter::context::ScratchBuffer;
use crate::filter::pixmap::FilterPixmap;
use crate::fine::FineKernel;
use crate::kurbo::{Affine, BezPath, Rect, Stroke, Vec2};
use crate::peniko::{BlendMode, Fill};
use alloc::vec::Vec;
use vello_common::clip::PushedClip;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::geometry::{PaddingU32, RectU16};
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;

/// Width and height of a window, the largest tile multiple in the `u16` domain.
const WINDOW_SIZE: i64 = 65532;

/// A rectangle in a filter's source space, which can exceed the `u16` domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bounds {
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
}

impl Bounds {
    const EMPTY: Self = Self::new(0, 0, 0, 0);

    const fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub(crate) fn of_viewport(width: u16, height: u16) -> Self {
        Self::new(0, 0, i64::from(width), i64::from(height))
    }

    pub(crate) fn of_rect(rect: RectU16, origin: (i64, i64)) -> Self {
        Self::new(
            i64::from(rect.x0) + origin.0,
            i64::from(rect.y0) + origin.1,
            i64::from(rect.x1) + origin.0,
            i64::from(rect.y1) + origin.1,
        )
    }

    fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    fn width(self) -> i64 {
        self.x1 - self.x0
    }

    fn height(self) -> i64 {
        self.y1 - self.y0
    }

    fn translate(self, (dx, dy): (i64, i64)) -> Self {
        Self::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }

    fn expand(self, padding: PaddingU32) -> Self {
        Self::new(
            self.x0 - i64::from(padding.left),
            self.y0 - i64::from(padding.top),
            self.x1 + i64::from(padding.right),
            self.y1 + i64::from(padding.bottom),
        )
    }

    fn intersect(self, other: Self) -> Self {
        Self::new(
            self.x0.max(other.x0),
            self.y0.max(other.y0),
            self.x1.min(other.x1),
            self.y1.min(other.y1),
        )
    }

    fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self::new(
            self.x0.min(other.x0),
            self.y0.min(other.y0),
            self.x1.max(other.x1),
            self.y1.max(other.y1),
        )
    }

    fn snap_to_tiles(self) -> Self {
        let (width, height) = (i64::from(Tile::WIDTH), i64::from(Tile::HEIGHT));
        Self::new(
            self.x0.div_euclid(width) * width,
            self.y0.div_euclid(height) * height,
            (self.x1 + width - 1).div_euclid(width) * width,
            (self.y1 + height - 1).div_euclid(height) * height,
        )
    }

    /// Convert bounds relative to `origin` into a `u16` rectangle, clipping the
    /// exclusive edge to the `u16` domain.
    fn to_rect_u16(self, origin: (i64, i64)) -> RectU16 {
        let local = self.translate((-origin.0, -origin.1));
        let narrow = |value: i64| u16::try_from(value.min(i64::from(u16::MAX))).unwrap();
        RectU16::new(
            narrow(local.x0),
            narrow(local.y0),
            narrow(local.x1),
            narrow(local.y1),
        )
    }
}

/// Return the region of a filter's source space that its output inside `view`
/// depends on. `view` is in the space of the filter's parent.
///
/// # Panics
///
/// Panics if the region, or the filter pixmap that can be allocated in it,
/// leaves the `u32` domain.
fn source_region(filter: &FilterData, view: Bounds) -> Bounds {
    let shift = filter.source_shift();
    let shift = (i64::from(shift.0), i64::from(shift.1));
    let (region, pixmap_end) = match filter.pure_offset() {
        Some((dx, dy)) => {
            let region = view.translate((shift.0 - dx, shift.1 - dy));
            (region, (region.x1, region.y1))
        }
        None => {
            let region = view.expand(filter.source_padding).translate(shift);
            let padding = filter.filter_padding;
            let end = (
                region.x1 + i64::from(padding.right),
                region.y1 + i64::from(padding.bottom),
            );
            (region, end)
        }
    };
    let pixmap_end = Bounds::new(0, 0, pixmap_end.0, pixmap_end.1).snap_to_tiles();
    assert!(
        region.x0 >= 0
            && region.y0 >= 0
            && pixmap_end.x1 <= i64::from(u32::MAX)
            && pixmap_end.y1 <= i64::from(u32::MAX),
        "filter source viewport exceeds u32 coordinate domain"
    );
    region
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
    /// The filter and its source region, in the filter's source space.
    pub(crate) filter: Option<(FilterData, Bounds)>,
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
    region: Bounds,
    /// Translation from root target coordinates to the filter's source space.
    root_shift: (i64, i64),
    /// Clips active when the layer was pushed.
    clips: Vec<Clip>,
    commands: Vec<Command>,
    /// Source regions of the layers open inside the recording, innermost last;
    /// `None` for layers without a filter.
    open_layers: Vec<Option<Bounds>>,
}

impl Recording {
    /// Start recording a filter layer whose output is visible in `view`.
    ///
    /// # Panics
    ///
    /// Panics before recording anything if the source region leaves the `u32` domain.
    pub(crate) fn new(
        filter: FilterData,
        view: Bounds,
        parent_root_shift: (i64, i64),
        clips: Vec<Clip>,
    ) -> Self {
        let region = source_region(&filter, view);
        let shift = filter.source_shift();
        Self {
            filter,
            region,
            root_shift: (
                parent_root_shift.0 + i64::from(shift.0),
                parent_root_shift.1 + i64::from(shift.1),
            ),
            clips,
            commands: Vec::new(),
            open_layers: Vec::new(),
        }
    }

    pub(crate) fn record(&mut self, command: Command) {
        self.commands.push(command);
    }

    /// Record a layer push. A nested filter's region is checked before anything is recorded.
    pub(crate) fn push_layer(&mut self, layer: Layer, filter: Option<FilterData>) {
        let enclosing = self
            .open_layers
            .iter()
            .rev()
            .find_map(|region| *region)
            .unwrap_or(self.region);
        let filter = filter.map(|filter| {
            let region = source_region(&filter, enclosing);
            (filter, region)
        });
        self.open_layers
            .push(filter.as_ref().map(|(_, region)| *region));
        self.record(Command::PushLayer(Layer { filter, ..layer }));
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
            self.region,
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

/// A filter layer rendered in windows, see the module documentation.
#[derive(Debug)]
pub(crate) struct WindowedFilter {
    filter: FilterData,
    windows: Vec<Window>,
    nested: Vec<Self>,
    /// Bounds of the filter pixmap in the filter's source space.
    pixmap_bounds: Bounds,
    /// Translation from the filter's source space to its parent's space.
    to_parent: (i64, i64),
}

impl WindowedFilter {
    fn new(
        commands: &[Command],
        filter: FilterData,
        region: Bounds,
        root_shift: (i64, i64),
        clips: &[Clip],
        level: Level,
    ) -> Self {
        let mut nested = Vec::new();
        let steps = plan(commands, root_shift, clips, level, &mut nested);
        let windows: Vec<Window> = window_grid(region)
            .map(|window| replay(window, &steps, &nested, root_shift, clips, level))
            .collect();
        let content = windows.iter().fold(Bounds::EMPTY, |content, window| {
            content.union(window.dispatcher.window_content(window.origin))
        });

        let shift = filter.source_shift();
        let shift = (-i64::from(shift.0), -i64::from(shift.1));
        let (pixmap_bounds, to_parent) = match filter.pure_offset() {
            Some((dx, dy)) => (content, (shift.0 + dx, shift.1 + dy)),
            None if content.is_empty() => (content, shift),
            None => {
                // Like a filter layer in a `u16` viewport, keep the pixmap inside
                // the non-negative source space so both decimate identically.
                let mut pixmap = content.expand(filter.filter_padding);
                pixmap.x0 = pixmap.x0.max(0);
                pixmap.y0 = pixmap.y0.max(0);
                (pixmap.snap_to_tiles(), shift)
            }
        };

        Self {
            filter,
            windows,
            nested,
            pixmap_bounds,
            to_parent,
        }
    }

    /// Bounds of the filtered pixmap in the parent's space.
    pub(crate) fn output_bounds(&self) -> Bounds {
        self.pixmap_bounds.translate(self.to_parent)
    }

    /// Return where the visible part of the output is composited in `view`, which
    /// starts at the origin of the parent's space.
    pub(crate) fn placement_in(&self, view: Bounds) -> RectU16 {
        let visible = self.output_bounds().intersect(view);
        if visible.is_empty() {
            return RectU16::ZERO;
        }
        visible.snap_to_tiles().to_rect_u16((0, 0))
    }

    /// Render and filter the layer.
    pub(crate) fn render<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> FilterOutput {
        let nested: Vec<FilterOutput> = self
            .nested
            .iter()
            .map(|filter| filter.render::<S, F>(simd, encoded_paints, image_resolver))
            .collect();
        let bounds = self.pixmap_bounds;
        let mut pixmap = FilterPixmap::new(
            u32::try_from(bounds.width()).unwrap(),
            u32::try_from(bounds.height()).unwrap(),
        );
        for window in &self.windows {
            if let Some((piece, rect)) = window.dispatcher.render_window::<S, F>(
                simd,
                encoded_paints,
                image_resolver,
                &nested,
            ) {
                let piece_bounds = Bounds::of_rect(rect, window.origin);
                copy_rows(
                    piece.data(),
                    piece_bounds,
                    &mut pixmap,
                    bounds,
                    piece_bounds,
                );
            }
        }
        if self.filter.pure_offset().is_none() && !bounds.is_empty() {
            F::filter_layer(
                &mut pixmap,
                PreparedFilter::new(&self.filter.filter, &self.filter.transform),
                &mut ScratchBuffer::new(),
            );
        }
        FilterOutput {
            pixmap,
            bounds: self.output_bounds(),
        }
    }
}

/// A filtered pixmap and its bounds in the parent's space.
#[derive(Debug)]
pub(crate) struct FilterOutput {
    pixmap: FilterPixmap,
    bounds: Bounds,
}

impl FilterOutput {
    /// Copy the output inside `dest`, a rectangle relative to `origin` in the
    /// parent's space, into a new pixmap that is transparent elsewhere.
    pub(crate) fn crop(&self, dest: RectU16, origin: (i64, i64)) -> Pixmap {
        let dest_bounds = Bounds::of_rect(dest, origin);
        let mut crop = FilterPixmap::new(u32::from(dest.width()), u32::from(dest.height()));
        let overlap = self.bounds.intersect(dest_bounds);
        if !overlap.is_empty() {
            copy_rows(
                self.pixmap.data(),
                self.bounds,
                &mut crop,
                dest_bounds,
                overlap,
            );
        }
        crop.into_pixmap()
    }
}

/// Copy the pixels inside `area` from a source covering `source_bounds` into a
/// pixmap covering `target_bounds`. All bounds are in the same space.
fn copy_rows(
    source: &[PremulRgba8],
    source_bounds: Bounds,
    target: &mut FilterPixmap,
    target_bounds: Bounds,
    area: Bounds,
) {
    let index = |bounds: Bounds, x: i64, y: i64| {
        usize::try_from((y - bounds.y0) * bounds.width() + x - bounds.x0).unwrap()
    };
    let width = usize::try_from(area.width()).unwrap();
    let pixels = target.data_mut();
    for y in area.y0..area.y1 {
        let from = index(source_bounds, area.x0, y);
        let to = index(target_bounds, area.x0, y);
        pixels[to..to + width].copy_from_slice(&source[from..from + width]);
    }
}

/// Replace nested filter layers in `commands` by steps that composite their
/// separately rendered output, which is appended to `nested`.
fn plan<'a>(
    commands: &'a [Command],
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
            && let Some((filter, region)) = &layer.filter
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
                *region,
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
    let start = Bounds::new(region.x0, region.y0, region.x0, region.y0).snap_to_tiles();
    let rows = (start.y0..region.y1).step_by(WINDOW_SIZE as usize);
    rows.flat_map(move |y| {
        (start.x0..region.x1)
            .step_by(WINDOW_SIZE as usize)
            .map(move |x| {
                Bounds::new(
                    x,
                    y,
                    (x + WINDOW_SIZE).min(region.x1),
                    (y + WINDOW_SIZE).min(region.y1),
                )
            })
    })
}

/// A filter that leaves its layer unchanged. Wrapping a window's content in it
/// records the tight bounds of that content and renders it into its own pixmap.
pub(crate) fn identity_filter() -> FilterData {
    FilterData::new(
        Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 }),
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
        (
            u32::try_from(origin.0).unwrap(),
            u32::try_from(origin.1).unwrap(),
        ),
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
    let dest = filter.output_bounds().intersect(window);
    let dest = if dest.is_empty() {
        RectU16::ZERO
    } else {
        dest.snap_to_tiles().to_rect_u16(origin)
    };
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
