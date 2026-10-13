// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The bounds of the strips a filter layer's commands draw, found from their
//! geometry as the recorder sees it, in work that follows the number of
//! lines drawn rather than their length.
//!
//! A path draws strips over the tiles its lines cross and the tiles between
//! them that its fill rule fills, so the bounds of its strips are those of
//! the tiles of its lines. The lines are those the recorder flattens the path
//! to, in a frame that a window can draw, and the tiles of a line are found
//! from a few of its rows, see [`TiledLine`]. A rectangle drawn on its own
//! covers the tiles [`vello_common::rect`] gives it.
//!
//! Clips cut what is drawn under them to the tiles they cover, and the
//! recorder culls what it flattens under them to the bounds of their strips,
//! which adds lines along the left edge of those bounds, see
//! `vello_common::fill_clip`. Rectangles that clip a rectangle cut it before
//! it is tiled; otherwise, where a clip covering a block of tiles cuts a
//! path, the tiles it keeps along the cut are found from the winding of the
//! path's lines, see [`filled_rows`]. Under clips of other shapes, what both
//! cover is found by drawing them in frames, but only where each covers some
//! tiles, see [`Extents::search`].
//!
//! A path larger than a frame is flattened in frames placed where its
//! segments reach furthest on each side, see [`Extents::far_path`].

use super::{Clip, Command, FilterGraph, Layer, Step, WINDOW_SIZE, stroke_reach};
use crate::filter::placed::{Bounds, Coord, Padding, Position};
use crate::kurbo::{
    Affine, BezPath, Line as Segment, ParamCurve, ParamCurveArclen, ParamCurveExtrema, PathSeg,
    Point, Rect, Stroke, StrokeCtx,
};
use crate::peniko::Fill;
#[cfg(not(feature = "std"))]
use crate::peniko::kurbo::common::FloatFuncs as _;
use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use vello_common::clip::{ClipRef, ClipShape, PathDataRef};
use vello_common::dash::{self, DASH_ACCURACY, Pattern, Phase, Subpaths, Walk};
use vello_common::fearless_simd::Level;
use vello_common::flatten::{self, FlattenCtx, Line};
use vello_common::geometry::RectU16;
use vello_common::strip::{StripFillSegment, visit_strip_fill_segments};
use vello_common::strip_generator::{StripGenerator, StripStorage};
use vello_common::tile::{MAX_LINES_PER_PATH, Tile};

/// The side of the square frames paths are flattened in, which windows draw.
const FRAME: i64 = WINDOW_SIZE;

/// How far inside a frame what is found in it must lie; lines that cross its
/// edges are tiled as a window would not, so tiles there are not counted.
const MARGIN: i64 = 64;

/// The width of a tile in pixels.
const TILE: i64 = Tile::WIDTH as i64;

/// The power of two [`TILE`] is.
const TILE_BITS: u32 = Tile::WIDTH.trailing_zeros();

const _: () = assert!(
    Tile::WIDTH == Tile::HEIGHT,
    "frames are measured in square tiles"
);

/// Something a filter layer draws or clips with, in root target coordinates.
#[derive(Clone, Copy, Debug)]
enum Item<'a> {
    /// A rectangle, mapped by the transform if it has one.
    Rect(Rect, Option<Affine>),
    Fill {
        path: &'a BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    },
    Stroke {
        path: &'a BezPath,
        stroke: &'a Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    },
}

impl<'a> Item<'a> {
    /// Return what `command` draws, if it draws.
    fn of_command(command: &'a Command) -> Option<Self> {
        match command {
            Command::FillPath {
                path,
                fill_rule,
                transform,
                aliasing_threshold,
                ..
            } => Some(Self::Fill {
                path,
                fill_rule: *fill_rule,
                transform: *transform,
                aliasing_threshold: *aliasing_threshold,
            }),
            Command::StrokePath {
                path,
                stroke,
                transform,
                aliasing_threshold,
                ..
            } => Some(Self::Stroke {
                path,
                stroke,
                transform: *transform,
                aliasing_threshold: *aliasing_threshold,
            }),
            Command::FillRect {
                rect, transform, ..
            } => Some(Self::Rect(*rect, Some(*transform))),
            _ => None,
        }
    }

    fn of_clip(clip: &'a Clip) -> Self {
        match clip {
            Clip::Path {
                path,
                fill_rule,
                transform,
                aliasing_threshold,
            } => Self::Fill {
                path,
                fill_rule: *fill_rule,
                transform: *transform,
                aliasing_threshold: *aliasing_threshold,
            },
            Clip::Rect(rect) => Self::Rect(*rect, None),
        }
    }

    /// Return the clip path of `layer`, if it has one.
    fn of_layer_clip(layer: &'a Layer) -> Option<Self> {
        layer.clip_path.as_ref().map(|path| Self::Fill {
            path,
            fill_rule: layer.fill_rule,
            transform: layer.clip_transform,
            aliasing_threshold: layer.aliasing_threshold,
        })
    }

    fn fill_rule(&self) -> Fill {
        match self {
            Self::Fill { fill_rule, .. } => *fill_rule,
            Self::Rect(..) | Self::Stroke { .. } => Fill::NonZero,
        }
    }

    /// Return the rectangle in root target coordinates, if the item is one.
    fn device_rect(&self) -> Option<Rect> {
        match self {
            Self::Rect(rect, transform) => {
                Some(transform.map_or(*rect, |transform| transform.transform_rect_bbox(*rect)))
            }
            Self::Fill { .. } | Self::Stroke { .. } => None,
        }
    }

    /// Return the rectangle in the frame `to_frame` maps root target
    /// coordinates to, mapped as the recorder maps it, if the item is one.
    fn frame_rect(&self, to_frame: Affine) -> Option<Rect> {
        match self {
            Self::Rect(rect, transform) => Some(
                transform
                    .map_or(to_frame, |transform| to_frame * transform)
                    .transform_rect_bbox(*rect),
            ),
            Self::Fill { .. } | Self::Stroke { .. } => None,
        }
    }

    /// Return how far a stroke's outline reaches from its path in device
    /// pixels, and zero for fills.
    fn reach(&self) -> f64 {
        match self {
            Self::Stroke {
                stroke, transform, ..
            } => stroke_reach(stroke, transform),
            Self::Rect(..) | Self::Fill { .. } => 0.0,
        }
    }

    /// Return a rectangle in root target coordinates that holds the lines the
    /// item is flattened to.
    fn device_bounds(&self) -> Rect {
        let (path, transform) = match self {
            Self::Rect(..) => return self.device_rect().unwrap().abs(),
            Self::Fill {
                path, transform, ..
            }
            | Self::Stroke {
                path, transform, ..
            } => (path, transform),
        };
        // Covers the flattening tolerance and what rounding adds.
        let reach = self.reach() + 1.0;
        transform
            .transform_rect_bbox(path.control_box())
            .inflate(reach, reach)
    }

    /// Return the dash pattern of a dashed stroke, unless dashing draws it
    /// whole.
    fn dashes(&self) -> Option<Pattern<'a>> {
        let Self::Stroke { stroke, .. } = self else {
            return None;
        };
        let entries = &stroke.dash_pattern[..];
        // Dashing walks a pattern with a negative entry back forever, see
        // `Pattern::new`.
        (!entries.is_empty() && entries.iter().all(|entry| *entry >= 0.0))
            .then(|| Pattern::new(stroke.dash_offset, entries))
    }

    /// Return the segments of the path in path coordinates as the recorder
    /// walks them, with the lines that close each subpath of a fill, see
    /// `flatten_simd::flatten`, and those of a stroke dashed with `dashes`
    /// each with the phase it starts at, see [`dash::phases`]; and the
    /// transform to root target coordinates.
    fn segments(&self, dashes: Option<&Pattern<'_>>) -> (Vec<(PathSeg, Option<Phase>)>, Affine) {
        let (path, transform, closes) = match self {
            Self::Rect(..) => return (Vec::new(), Affine::IDENTITY),
            Self::Fill {
                path, transform, ..
            } => (path, transform, true),
            Self::Stroke {
                path, transform, ..
            } => (path, transform, false),
        };
        if let Some(dashes) = dashes {
            let segments = dash::phases(path.iter(), dashes);
            return (
                segments.map(|(seg, phase)| (seg, Some(phase))).collect(),
                *transform,
            );
        }
        let mut segments = Vec::new();
        let mut subpaths = Subpaths::default();
        for element in path.iter() {
            let opening = subpaths.open();
            match subpaths.walk(element) {
                Walk::Move => segments.extend(opening.filter(|_| closes)),
                Walk::Segment(segment) | Walk::Close(Some(segment)) => segments.push(segment),
                Walk::Close(None) => {}
            }
        }
        segments.extend(subpaths.open().filter(|_| closes));
        (
            segments.into_iter().map(|seg| (seg, None)).collect(),
            *transform,
        )
    }
}

/// Tiles `x0..=x1` by `y0..=y1` of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tiles {
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
}

impl Tiles {
    fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Option<Self> {
        (x0 <= x1 && y0 <= y1).then_some(Self { x0, y0, x1, y1 })
    }

    fn intersect(self, other: Self) -> Option<Self> {
        Self::new(
            self.x0.max(other.x0),
            self.y0.max(other.y0),
            self.x1.min(other.x1),
            self.y1.min(other.y1),
        )
    }

    fn union(tiles: Option<Self>, other: Option<Self>) -> Option<Self> {
        match (tiles, other) {
            (Some(a), Some(b)) => Some(Self {
                x0: a.x0.min(b.x0),
                y0: a.y0.min(b.y0),
                x1: a.x1.max(b.x1),
                y1: a.y1.max(b.y1),
            }),
            (tiles, None) => tiles,
            (None, other) => other,
        }
    }

    /// Return the tiles of `bounds` in the frame at `origin`, clamped to the
    /// `i64` domain, which frames lie far inside.
    fn of_bounds(bounds: &Bounds, origin: &Position) -> Option<Self> {
        if bounds.is_empty() {
            return None;
        }
        let bounds = bounds.snap_to_tiles();
        let tile = |value: &Coord, origin: &Coord| (value - origin).clamped_i64() / TILE;
        Self::new(
            tile(&bounds.x0, &origin.0),
            tile(&bounds.y0, &origin.1),
            tile(&bounds.x1, &origin.0) - 1,
            tile(&bounds.y1, &origin.1) - 1,
        )
    }

    /// Return the pixels of the tiles in root target coordinates.
    fn to_bounds(self, origin: &Position) -> Bounds {
        Bounds::new(
            &origin.0 + self.x0 * TILE,
            &origin.1 + self.y0 * TILE,
            &origin.0 + (self.x1 + 1) * TILE,
            &origin.1 + (self.y1 + 1) * TILE,
        )
    }
}

/// How the tiler covers a line with tiles, see
/// `Tiles::make_tiles_analytic_aa_impl`: on each row the line crosses, the
/// columns between where it enters and leaves the row, in tile units of
/// `f32` as the tiler computes them, in a viewport that holds the line.
#[derive(Clone, Copy, Debug)]
struct TiledLine {
    p0_x: f32,
    p0_y: f32,
    x_slope: f32,
    top_x: f32,
    bottom_x: f32,
    bottom_y: f32,
    /// The first row and the row after the last.
    rows: (i64, i64),
    /// The column of a line inside one column of tiles.
    column: Option<i64>,
}

impl TiledLine {
    /// Return the line, unless the tiler gives it no tiles.
    fn new(line: Line) -> Option<Self> {
        let tile = f32::from(Tile::WIDTH);
        let (p0_x, p0_y) = (line.p0.x / tile, line.p0.y / tile);
        let (p1_x, p1_y) = (line.p1.x / tile, line.p1.y / tile);
        let (left, right) = if p0_x < p1_x {
            (p0_x, p1_x)
        } else {
            (p1_x, p0_x)
        };
        let (top_y, top_x, bottom_y, bottom_x) = if p0_y < p1_y {
            (p0_y, p0_x, p1_y, p1_x)
        } else {
            (p1_y, p1_x, p0_y, p0_x)
        };
        let rows = (whole(top_y.floor()), whole(bottom_y.ceil()));
        if rows.0 >= rows.1 {
            return None;
        }
        let same_tile = top_y.floor() == bottom_y.floor() && top_x.floor() == bottom_x.floor();
        Some(Self {
            p0_x,
            p0_y,
            x_slope: (p1_x - p0_x) / (p1_y - p0_y),
            top_x,
            bottom_x,
            bottom_y,
            rows,
            column: (same_tile || left == right).then(|| whole(left.floor())),
        })
    }

    /// Return the columns the line covers on `row`, which it crosses.
    fn columns(&self, row: i64) -> (i64, i64) {
        if let Some(column) = self.column {
            return (column, column);
        }
        let at = |y: f32| self.p0_x + (y - self.p0_y) * self.x_slope;
        let y = row as f32;
        let (a, b) = if row == self.rows.0 {
            let bottom = (y + 1.0).min(self.bottom_y);
            let b = if bottom == self.bottom_y {
                self.bottom_x
            } else {
                at(bottom)
            };
            (self.top_x, b)
        } else {
            let b = if self.bottom_y < y + 1.0 {
                self.bottom_x
            } else {
                at(y + 1.0)
            };
            (at(y), b)
        };
        (whole(a.min(b).floor()), whole(a.max(b).floor()))
    }

    /// Return the tiles of the line inside `region`.
    ///
    /// The rows between the first and the last enter and leave at `f32`
    /// positions interpolated along the line, which move monotonically, so
    /// the rows that meet the columns of `region` are found by bisection.
    fn within(&self, region: Tiles) -> Option<Tiles> {
        let rows = (self.rows.0.max(region.y0), (self.rows.1 - 1).min(region.y1));
        if rows.0 > rows.1 {
            return None;
        }
        let on_row = |row: i64| {
            let (x0, x1) = self.columns(row);
            Tiles::new(x0.max(region.x0), row, x1.min(region.x1), row)
        };
        let mut tiles = None;
        for row in [self.rows.0, self.rows.1 - 1] {
            if (rows.0..=rows.1).contains(&row) {
                tiles = Tiles::union(tiles, on_row(row));
            }
        }
        let middle = ((self.rows.0 + 1).max(rows.0), (self.rows.1 - 2).min(rows.1));
        if middle.0 > middle.1 {
            return tiles;
        }
        // Along the rows, both ends of the covered columns move one way.
        let rising = self.column.is_some() || self.x_slope >= 0.0;
        let reaches = |row: i64| self.columns(row).1 >= region.x0;
        let starts = |row: i64| self.columns(row).0 <= region.x1;
        let (first, last) = if rising {
            (first_true(middle, reaches), last_true(middle, starts))
        } else {
            (first_true(middle, starts), last_true(middle, reaches))
        };
        if let (Some(first), Some(last)) = (first, last)
            && first <= last
        {
            let (left, right) = if rising { (first, last) } else { (last, first) };
            tiles = Tiles::union(
                tiles,
                Tiles::new(
                    self.columns(left).0.max(region.x0),
                    first,
                    self.columns(right).1.min(region.x1),
                    last,
                ),
            );
        }
        tiles
    }
}

/// Return `value`, a whole `f32` in a frame, as an integer.
fn whole(value: f32) -> i64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "frames hold far fewer than 2^63 tiles; `as` saturates"
    )]
    let value = value as i64;
    value
}

/// Return the first of `rows` for which `holds`, which holds for every row
/// after one for which it holds.
fn first_true(rows: (i64, i64), holds: impl Fn(i64) -> bool) -> Option<i64> {
    let (mut low, mut high) = rows;
    if !holds(high) {
        return None;
    }
    while low < high {
        let middle = low + (high - low) / 2;
        if holds(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Some(low)
}

/// Return the last of `rows` for which `holds`, which holds for every row
/// before one for which it holds.
fn last_true(rows: (i64, i64), holds: impl Fn(i64) -> bool) -> Option<i64> {
    let (mut low, mut high) = rows;
    if !holds(low) {
        return None;
    }
    while low < high {
        let middle = low + (high - low + 1) / 2;
        if holds(middle) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Some(low)
}

/// Return the first and last of `rows` of tiles in which the winding of
/// `lines` along the middle of `column` is filled under `fill_rule`.
///
/// Where no line crosses a tile, the winding is that of its middle, so every
/// such row holds a tile that the clip cut through `column` keeps, either one
/// the fill rule fills or one a line crosses.
fn filled_rows(
    lines: &[Line],
    column: i64,
    rows: (i64, i64),
    fill_rule: Fill,
) -> Option<(i64, i64)> {
    let x = (column as f64 + 0.5) * TILE as f64;
    // Each line counts where it lies left of the middle of the column.
    let mut changes: Vec<(f64, i32)> = Vec::new();
    for line in lines {
        let (p0, p1) = (
            (f64::from(line.p0.x), f64::from(line.p0.y)),
            (f64::from(line.p1.x), f64::from(line.p1.y)),
        );
        if p0.1 == p1.1 {
            continue;
        }
        let direction = if p0.1 < p1.1 { 1 } else { -1 };
        let (top, bottom) = if p0.1 < p1.1 { (p0, p1) } else { (p1, p0) };
        let (start, end) = match (top.0 < x, bottom.0 < x) {
            (true, true) => (top.1, bottom.1),
            (false, false) => continue,
            (left_at_top, _) => {
                let cross = top.1 + (x - top.0) * (bottom.1 - top.1) / (bottom.0 - top.0);
                if left_at_top {
                    (top.1, cross)
                } else {
                    (cross, bottom.1)
                }
            }
        };
        if start < end {
            changes.push((start, direction));
            changes.push((end, -direction));
        }
    }
    changes.sort_by(|a, b| a.0.total_cmp(&b.0));
    let filled = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };
    let (top, bottom) = ((rows.0 * TILE) as f64, ((rows.1 + 1) * TILE) as f64);
    let mut found: Option<(i64, i64)> = None;
    let mut winding = 0;
    for (index, &(y, change)) in changes.iter().enumerate() {
        winding += change;
        let next = changes.get(index + 1).map_or(f64::INFINITY, |next| next.0);
        let (from, to) = (y.max(top), next.min(bottom));
        if from < to && filled(winding) {
            let span = (
                whole_f64((from / TILE as f64).floor()),
                whole_f64((to / TILE as f64).ceil()) - 1,
            );
            found = Some(found.map_or(span, |(first, last)| (first.min(span.0), last.max(span.1))));
        }
    }
    found
}

/// Return `value`, a whole `f64` in a frame, as an integer.
fn whole_f64(value: f64) -> i64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "frames hold far fewer than 2^63 tiles; `as` saturates"
    )]
    let value = value as i64;
    value
}

/// Return the start of the tile holding the pixel `value`.
fn tile_floor(value: &Coord) -> Coord {
    value.shr(TILE_BITS).shl(TILE_BITS)
}

/// What the clips in effect cover.
#[derive(Clone, Debug)]
enum Cover {
    All,
    /// Rectangles, whose intersection cuts a rectangle drawn under them
    /// before it is tiled, and anything else to its tiles.
    Rect(Rect),
    /// Clips covering every tile of the bounds, which is not known to be a
    /// rectangle, so a rectangle drawn under them is cut to the bounds.
    Tiles(Bounds),
    /// Clips of other shapes, whose strips lie in `all`, and in the
    /// recorder's viewport in `home`.
    Paths {
        all: Bounds,
        home: Bounds,
    },
}

/// The viewport the recorder draws a filter layer in, see
/// `ViewportState::push_root_viewport`, or the part of it at its origin that
/// a strip generator addresses if it is larger, where what is drawn is found
/// in the recorder's coordinates.
///
/// What lies inside it a tile away from its edges is found from its geometry
/// there; anything else, which the recorder cuts at its edges, and what clip
/// paths cut, is searched for as outside of it, flattened in its frame, which
/// holds any part of it, and drawn in windows of it, see
/// `StripGenerator::new_window`.
struct Home {
    origin: Position,
    /// The width and height of the viewport.
    size: (u16, u16),
    /// The viewport, which culls what is drawn in it without clips.
    viewport: Bounds,
    /// The tiles of the viewport, outside of which nothing is drawn in it.
    tiles: Bounds,
    /// The tiles a tile away from its edges.
    inside: Bounds,
    to_frame: Affine,
}

impl Home {
    fn new(viewport: &Bounds) -> Option<Self> {
        if viewport.is_empty() {
            return None;
        }
        let side = |length: Coord| {
            length
                .to_i64()
                .and_then(|length| u16::try_from(length).ok())
                .unwrap_or(FRAME as u16)
        };
        let size = (side(viewport.width()), side(viewport.height()));
        let origin = viewport.origin();
        let viewport = Bounds::new(
            origin.0.clone(),
            origin.1.clone(),
            &origin.0 + i64::from(size.0),
            &origin.1 + i64::from(size.1),
        );
        let tiles = viewport.snap_to_tiles();
        let inside = Bounds::new(
            &tiles.x0 + TILE,
            &tiles.y0 + TILE,
            &tiles.x1 - TILE,
            &tiles.y1 - TILE,
        );
        Some(Self {
            to_frame: Affine::translate((-origin.0.to_f64(), -origin.1.to_f64())),
            origin,
            size,
            viewport,
            tiles,
            inside,
        })
    }

    /// Return the window of the viewport that holds `area`, which lies in
    /// its tiles, in the viewport's coordinates.
    fn window(&self, area: &Bounds) -> RectU16 {
        let area = area.snap_to_tiles();
        let local = |value: &Coord, origin: &Coord, size: u16| {
            u16::try_from((value - origin).clamped_i64().clamp(0, size.into())).unwrap()
        };
        RectU16::new(
            local(&area.x0, &self.origin.0, self.size.0),
            local(&area.y0, &self.origin.1, self.size.1),
            local(&area.x1, &self.origin.0, self.size.0),
            local(&area.y1, &self.origin.1, self.size.1),
        )
    }
}

/// Finds the bounds of the strips the commands of a filter layer draw.
pub(super) struct Extents<'a> {
    level: Level,
    /// The viewport that bounds commands drawing without bounds, see `Node`.
    viewport: &'a Bounds,
    home: Option<Home>,
    lines: Vec<Line>,
    flatten: FlattenCtx,
    stroke: StrokeCtx,
}

impl<'a> Extents<'a> {
    pub(super) fn new(level: Level, viewport: &'a Bounds) -> Self {
        Self {
            level,
            viewport,
            home: Home::new(viewport),
            lines: Vec::new(),
            flatten: FlattenCtx::default(),
            stroke: StrokeCtx::default(),
        }
    }

    /// Return the bounds of the content of the source of layer `index`, with
    /// the nested layers composited where `pixmap` says, as the recorder
    /// bounds a layer's content, see `CommandRecorder::pop_layer_placed`.
    pub(super) fn content(
        &mut self,
        graph: &FilterGraph,
        index: usize,
        pixmap: &dyn Fn(usize) -> Bounds,
    ) -> Bounds {
        let layer = &graph.layers[index];
        let mut clips: Vec<&Clip> = Vec::new();
        let mut covers = vec![Cover::All];
        for clip in &layer.clips {
            let cover = self.push_clip(&covers, &clips, clip);
            covers.push(cover);
            clips.push(clip);
        }
        // The content of each open layer, and the bounds of its clip path.
        let mut open: Vec<(Bounds, Option<Bounds>)> = vec![(Bounds::EMPTY, None)];
        let add = |open: &mut Vec<(Bounds, Option<Bounds>)>, bounds: &Bounds| {
            let content = &mut open.last_mut().unwrap().0;
            *content = content.union(bounds);
        };
        for step in &layer.steps {
            let command = match *step {
                Step::Command { command, .. } => &graph.commands[command],
                Step::Nested { layer, .. } => {
                    add(&mut open, &pixmap(layer));
                    continue;
                }
            };
            match command {
                Command::PushClip(clip) => {
                    let cover = self.push_clip(&covers, &clips, clip);
                    covers.push(cover);
                    clips.push(clip);
                }
                Command::PopClip => {
                    if clips.pop().is_some() {
                        covers.pop();
                    }
                }
                Command::PushLayer(pushed) => {
                    let clip =
                        Item::of_layer_clip(pushed).map(|item| self.extent(&item, &covers, &clips));
                    open.push((Bounds::EMPTY, clip));
                }
                Command::PopLayer => {
                    let (content, clip) = open.pop().unwrap();
                    let content = clip.map_or(content.clone(), |clip| content.intersect(&clip));
                    add(&mut open, &content);
                }
                _ => {
                    if let Some(item) = Item::of_command(command) {
                        let extent = self.extent(&item, &covers, &clips);
                        add(&mut open, &extent);
                    }
                }
            }
        }
        open.swap_remove(0).0
    }

    /// Push `clip` onto `clips`, which cover the last of `covers`, and return
    /// what the clips cover then, see `ClipContext::push_clip_rect`.
    fn push_clip(&mut self, covers: &[Cover], clips: &[&Clip], clip: &Clip) -> Cover {
        let item = Item::of_clip(clip);
        let cover = match (covers.last().unwrap(), clip) {
            (Cover::All, Clip::Rect(rect)) => Cover::Rect(self.unbounded_cut(rect.abs())),
            (Cover::Rect(clip), Clip::Rect(rect)) => Cover::Rect(rect.abs().intersect(*clip)),
            (Cover::Tiles(tiles), Clip::Rect(rect)) => {
                let rect = rect.abs().intersect(to_rect(tiles));
                Cover::Tiles(self.rect_tiles(rect).intersect(tiles))
            }
            (Cover::All | Cover::Rect(_) | Cover::Tiles(_), Clip::Path { .. })
                if is_axis_aligned_rect(&item) =>
            {
                Cover::Tiles(self.outside(&item, covers, clips))
            }
            _ => Cover::Paths {
                all: self.outside(&item, covers, clips),
                home: Bounds::EMPTY,
            },
        };
        let drawn = self.home_extent(&item, covers, clips);
        match cover {
            Cover::Tiles(bounds) => Cover::Tiles(bounds.union(&drawn)),
            Cover::Paths { all, .. } => Cover::Paths {
                all: all.union(&drawn),
                home: drawn,
            },
            Cover::All | Cover::Rect(_) => cover,
        }
    }

    /// Return the bounds of the strips of the clips that cover `cover`,
    /// which the recorder culls what it flattens under them to, see
    /// `StripGenerator::generate_filled_path`, or `None` without clips.
    fn clip_bounds(&self, cover: &Cover) -> Option<Bounds> {
        match cover {
            Cover::All => None,
            Cover::Rect(rect) => Some(self.rect_tiles(*rect)),
            Cover::Tiles(bounds) | Cover::Paths { all: bounds, .. } => Some(bounds.clone()),
        }
    }

    /// Return what the recorder culls what it flattens to where it draws
    /// `area`: in its viewport, the bounds of the strips the clips that
    /// cover `cover` have there, or the viewport without clips, and
    /// elsewhere those of their strips anywhere, see [`Self::clip_bounds`].
    fn cull(&self, cover: &Cover, area: &Bounds) -> Option<Bounds> {
        let Some(home) = self.home_holding(area) else {
            return self.clip_bounds(cover);
        };
        Some(match cover {
            Cover::All => home.viewport.clone(),
            // The strips of rectangles and blocks of tiles cut to the viewport.
            Cover::Rect(_) | Cover::Tiles(_) => {
                self.clip_bounds(cover).unwrap().intersect(&home.tiles)
            }
            Cover::Paths { home: bounds, .. } => bounds.clone(),
        })
    }

    /// Return a rectangle without bounds cut to the viewport the recorder
    /// draws it in, as `StripGenerator::generate_filled_rect_fast` does.
    fn unbounded_cut(&self, rect: Rect) -> Rect {
        let finite = [rect.x0, rect.y0, rect.x1, rect.y1]
            .iter()
            .all(|value| value.is_finite());
        if finite {
            rect
        } else {
            rect.intersect(to_rect(self.viewport))
        }
    }

    /// Return the bounds of the strips `item` draws under `clips`, which
    /// cover the last of `covers`, the first of which is that of no clip and
    /// each next that of one more of `clips`: in the recorder's viewport as
    /// the recorder draws it, and outside of it from its geometry.
    fn extent(&mut self, item: &Item<'_>, covers: &[Cover], clips: &[&Clip]) -> Bounds {
        let drawn = self.home_extent(item, covers, clips);
        drawn.union(&self.outside(item, covers, clips))
    }

    /// Return the bounds of the strips `item` draws under `clips` in the
    /// recorder's viewport, see [`Home`].
    fn home_extent(&mut self, item: &Item<'_>, covers: &[Cover], clips: &[&Clip]) -> Bounds {
        let Some(home) = &self.home else {
            return Bounds::EMPTY;
        };
        let cover = covers.last().unwrap();
        // What the recorder culls to in its viewport.
        let cull = self.cull(cover, &home.tiles).unwrap();
        let Some(reach) = self.reach(item, Some(&cull)) else {
            return Bounds::EMPTY;
        };
        let area = reach.intersect(&home.tiles);
        if area.is_empty() {
            return Bounds::EMPTY;
        }
        if !home.inside.contains(&reach) || matches!(cover, Cover::Paths { .. }) {
            let area = match cover {
                Cover::All => area,
                Cover::Rect(_) | Cover::Tiles(_) | Cover::Paths { .. } => area.intersect(&cull),
            };
            return self.search(item, covers, clips, area);
        }
        let (origin, to_frame, size) = (home.origin.clone(), home.to_frame, home.size);
        // The clips' strips in the viewport.
        let cull = match cover {
            Cover::All => None,
            Cover::Rect(_) | Cover::Tiles(_) | Cover::Paths { .. } => Some(cull),
        };
        if let Some(rect) = item.frame_rect(to_frame) {
            // As `StripGenerator::generate_filled_rect_fast` cuts it.
            let viewport = Rect::new(0.0, 0.0, f64::from(size.0), f64::from(size.1));
            let rect = rect.abs().intersect(viewport);
            let rect = match cover {
                Cover::All => rect,
                Cover::Rect(clip) => rect.intersect(to_frame.transform_rect_bbox(*clip)),
                Cover::Tiles(_) | Cover::Paths { .. } => {
                    let cull = cull.as_ref().unwrap();
                    rect.intersect(to_frame.transform_rect_bbox(to_rect(cull)))
                }
            };
            let tiles = frame_rect_tiles(rect, &origin);
            return match cull {
                Some(cull) if !matches!(cover, Cover::Rect(_)) => tiles.intersect(&cull),
                _ => tiles,
            };
        }
        let cut = match cover {
            Cover::All => None,
            Cover::Rect(_) | Cover::Tiles(_) | Cover::Paths { .. } => cull.clone(),
        };
        if cut.as_ref().is_some_and(Bounds::is_empty) {
            return Bounds::EMPTY;
        }
        self.flatten(item, &origin, cull.as_ref());
        let area = cut
            .as_ref()
            .map_or(reach.clone(), |cut| reach.intersect(cut));
        let Some(region) = Tiles::of_bounds(&area, &origin) else {
            return Bounds::EMPTY;
        };
        let cut = cut.and_then(|cut| Tiles::of_bounds(&cut, &origin));
        frame_tiles(&self.lines, region, cut, item.fill_rule())
            .map_or(Bounds::EMPTY, |tiles| tiles.to_bounds(&origin))
    }

    /// Return bounds holding the tiles `item` can cover, flattened with the
    /// culling of clips whose strips lie in `cull`, from its geometry, or
    /// `None` if it covers none.
    fn reach(&self, item: &Item<'_>, cull: Option<&Bounds>) -> Option<Bounds> {
        let reach = match item.device_rect() {
            Some(rect) => Bounds::covering(self.unbounded_cut(rect.abs())),
            None => Some(
                Bounds::covering(item.device_bounds())
                    .map_or_else(|| self.viewport.clone(), |bounds| culled(bounds, cull)),
            ),
        };
        // A rectangle covers the tile holding its right and bottom edges.
        reach.map(|reach| reach.expand(&Padding::uniform(TILE.into())).snap_to_tiles())
    }

    /// Return the bounds of the strips `item` draws under `clips` outside the
    /// recorder's viewport, see [`Self::extent`].
    fn outside(&mut self, item: &Item<'_>, covers: &[Cover], clips: &[&Clip]) -> Bounds {
        let cull = self.clip_bounds(covers.last().unwrap());
        let Some(reach) = self.reach(item, cull.as_ref()) else {
            return Bounds::EMPTY;
        };
        let pieces = match &self.home {
            Some(home) => outside_of(&reach, &home.tiles),
            None => vec![reach],
        };
        pieces.iter().fold(Bounds::EMPTY, |found, piece| {
            found.union(&self.extent_in(item, covers, clips, piece))
        })
    }

    /// Return the bounds of the strips `item` draws under `clips` inside
    /// `piece`, from its geometry, see [`Self::extent`].
    fn extent_in(
        &mut self,
        item: &Item<'_>,
        covers: &[Cover],
        clips: &[&Clip],
        piece: &Bounds,
    ) -> Bounds {
        let cover = covers.last().unwrap();
        let cull = self.clip_bounds(cover);
        match (item.device_rect(), cover) {
            (Some(rect), Cover::All) => self
                .rect_tiles(self.unbounded_cut(rect.abs()))
                .intersect(piece),
            (Some(rect), Cover::Rect(clip)) => self
                .rect_tiles(rect.abs().intersect(*clip))
                .intersect(piece),
            (Some(rect), Cover::Tiles(tiles)) => self
                .rect_tiles(rect.abs().intersect(to_rect(tiles)))
                .intersect(tiles)
                .intersect(piece),
            (None, Cover::All) => self.path(item, Some(piece), None),
            (None, Cover::Rect(_) | Cover::Tiles(_)) => {
                let cut = cull.as_ref().unwrap().intersect(piece);
                self.path(item, Some(&cut), cull.as_ref())
            }
            (_, Cover::Paths { all: bounds, .. }) => {
                let area = self.covered(item, Some(&bounds.intersect(piece)), cull.as_ref());
                self.search(item, covers, clips, area)
            }
        }
    }

    /// Return the bounds of the strips `item` draws inside `cut`, a block of
    /// tiles a clip covers, or anywhere if there is none, flattened with the
    /// culling of clips whose strips lie in `cull`.
    fn covered(&mut self, item: &Item<'_>, cut: Option<&Bounds>, cull: Option<&Bounds>) -> Bounds {
        match item.device_rect() {
            Some(rect) => {
                let tiles = self.rect_tiles(self.unbounded_cut(rect.abs()));
                cut.map_or(tiles.clone(), |cut| tiles.intersect(cut))
            }
            None => self.path(item, cut, cull),
        }
    }

    /// Return the tiles a rectangle covers, in root target coordinates, as
    /// `vello_common::rect::render` covers it in `f32` coordinates of its
    /// viewport: from the tile holding its top left pixel to the one holding
    /// its right edge and its bottom pixel.
    ///
    /// Each side is converted in the frame of the viewport if it lies there,
    /// as the recorder converts it, and otherwise in a frame at the tile it
    /// lies in.
    fn rect_tiles(&self, rect: Rect) -> Bounds {
        let finite = [rect.x0, rect.y0, rect.x1, rect.y1]
            .iter()
            .all(|value| value.is_finite());
        if !finite || rect.is_zero_area() {
            return Bounds::EMPTY;
        }
        let viewport = self.viewport.origin();
        let pixel = |value: f64, origin: &Coord, round: fn(f32) -> f32| {
            let origin = if (0.0..FRAME as f64).contains(&(value - origin.to_f64())) {
                origin.clone()
            } else {
                tile_floor(&Coord::floor_of(value))
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the recorder converts coordinates to `f32`"
            )]
            let local = (value - origin.to_f64()) as f32;
            Coord::from(whole(round(local))) + origin
        };
        Bounds::new(
            tile_floor(&pixel(rect.x0, &viewport.0, f32::floor)),
            tile_floor(&pixel(rect.y0, &viewport.1, f32::floor)),
            tile_floor(&pixel(rect.x1, &viewport.0, f32::floor)) + TILE,
            -tile_floor(&-pixel(rect.y1, &viewport.1, f32::ceil)),
        )
    }

    /// Return the bounds of the tiles a path covers inside `cut`, a block of
    /// tiles a clip covers, or anywhere if there is none, flattened with the
    /// culling of clips whose strips lie in `cull`.
    fn path(&mut self, item: &Item<'_>, cut: Option<&Bounds>, cull: Option<&Bounds>) -> Bounds {
        if cull.is_some_and(Bounds::is_empty) {
            return Bounds::EMPTY;
        }
        let device = item.device_bounds();
        let Some(bounds) = Bounds::covering(device) else {
            return self.unbounded_path(item, cut, cull);
        };
        let bounds = culled(bounds, cull);
        let area = cut.map_or(bounds.clone(), |cut| bounds.intersect(cut));
        if area.is_empty() {
            return Bounds::EMPTY;
        }
        if !fits(&area) && self.home_holding(&area).is_none() {
            return self.far_path(item, &area, cut, cull);
        }
        let origin = self.frame_at(&area);
        self.flatten(item, &origin, cull);
        let region = Tiles::of_bounds(&area, &origin).unwrap();
        let cut = cut.and_then(|cut| Tiles::of_bounds(cut, &origin));
        frame_tiles(&self.lines, region, cut, item.fill_rule())
            .map_or(Bounds::EMPTY, |tiles| tiles.to_bounds(&origin))
    }

    /// Return the recorder's viewport if `area` lies in its tiles, see
    /// [`Home`].
    fn home_holding(&self, area: &Bounds) -> Option<&Home> {
        self.home.as_ref().filter(|home| home.tiles.contains(area))
    }

    /// Return the origin of a frame holding `area` inside its margin: that of
    /// the viewport if it does, or if `area` lies in the recorder's viewport,
    /// where the recorder flattens, and otherwise one at the top left of
    /// `area`.
    fn frame_at(&self, area: &Bounds) -> Position {
        if let Some(home) = self.home_holding(area) {
            return home.origin.clone();
        }
        let origin = self.viewport.origin();
        let frame = Bounds::new(
            &origin.0 + MARGIN,
            &origin.1 + MARGIN,
            &origin.0 + (FRAME - MARGIN),
            &origin.1 + (FRAME - MARGIN),
        );
        if frame.contains(area) {
            return origin;
        }
        let area = area.snap_to_tiles();
        (&area.x0 - MARGIN, &area.y0 - MARGIN)
    }

    /// Flatten `item` into `self.lines` in the frame at `origin`, as the
    /// recorder flattens it in a viewport there under clips whose strips lie
    /// in `cull`, see `StripGenerator`.
    fn flatten(&mut self, item: &Item<'_>, origin: &Position, cull: Option<&Bounds>) {
        let level = self.level;
        let to_frame = Affine::translate((-origin.0.to_f64(), -origin.1.to_f64()));
        let cull = frame_cull(cull, origin);
        match item {
            Item::Fill {
                path, transform, ..
            } => flatten::fill(
                level,
                path.iter(),
                to_frame * *transform,
                &mut self.lines,
                &mut self.flatten,
                cull,
            ),
            Item::Stroke {
                path,
                stroke,
                transform,
                ..
            } => flatten::stroke(
                level,
                path.iter(),
                stroke,
                to_frame * *transform,
                &mut self.lines,
                &mut self.flatten,
                &mut self.stroke,
                cull,
            ),
            Item::Rect(..) => unreachable!("rectangles are not flattened"),
        }
        self.lines.truncate(MAX_LINES_PER_PATH as usize);
    }

    /// Return the bounds of the tiles a path whose bounds are not finite
    /// covers inside `cut`.
    ///
    /// The recorder ignores a path whose device coordinates are not finite,
    /// see `flatten::fill`, so it draws nothing in any frame.
    fn unbounded_path(
        &mut self,
        item: &Item<'_>,
        cut: Option<&Bounds>,
        cull: Option<&Bounds>,
    ) -> Bounds {
        let origin = self.viewport.origin();
        self.flatten(item, &origin, cull);
        if self.lines.is_empty() {
            return Bounds::EMPTY;
        }
        // Not reached by the paths the flattener accepts; the viewport bounds
        // what the recorder draws.
        let viewport = self.viewport.clone();
        cut.map_or(viewport.clone(), |cut| viewport.intersect(cut))
    }

    /// Return the bounds of the tiles a path larger than a frame covers in
    /// `area`, which holds them and lies in `cut` if there is one.
    ///
    /// The tiles reaching furthest on a side are those of the lines near the
    /// points of the path that reach furthest there inside `area`: where its
    /// segments do, where they cross the edges of `area`, the corners of
    /// `cut`, which a fill can cover, and where what lies left of the clips'
    /// strips is drawn along their left edge, see `fill_clip`, the points
    /// left of it moved onto it. On a dashed stroke, where the dash or gap
    /// such a point lies in starts and ends: the drawn points of a gap that
    /// reach furthest on either side of it are there, and so are those where
    /// dashing draws a line from one dash to another, see [`dash::phases`].
    /// From the furthest,
    /// the path is flattened in a frame around each point not yet in one,
    /// with that side of the frame beyond the point by a stroke's reach,
    /// until no point is left that reaches further than the tiles found.
    fn far_path(
        &mut self,
        item: &Item<'_>,
        area: &Bounds,
        cut: Option<&Bounds>,
        cull: Option<&Bounds>,
    ) -> Bounds {
        let reach = item.reach();
        let limits = to_rect(area);
        let reached = limits.inflate(reach, reach);
        let corners = [
            Point::new(limits.x0, limits.y0),
            Point::new(limits.x1, limits.y0),
            Point::new(limits.x0, limits.y1),
            Point::new(limits.x1, limits.y1),
        ];
        let cull_rect = cull.map(to_rect);
        let mut edges = vec![
            Segment::new(corners[0], corners[1]),
            Segment::new(corners[2], corners[3]),
            Segment::new(corners[0], corners[2]),
            Segment::new(corners[1], corners[3]),
        ];
        if let Some(cull) = cull_rect {
            edges.push(Segment::new(
                Point::new(cull.x0, reached.y0),
                Point::new(cull.x0, reached.y1),
            ));
        }
        let dashes = item.dashes();
        let (segments, transform) = item.segments(dashes.as_ref());
        let mut points: Vec<Point> = Vec::new();
        for (segment, phase) in segments {
            let device = transform * segment;
            if !overlap(device.bounding_box(), reached) {
                continue;
            }
            // Each parameter, with the point on an edge of `area` it crosses
            // there.
            let mut ends: Vec<(f64, Option<Point>)> = [0.0, 1.0]
                .into_iter()
                .chain(device.extrema())
                .map(|t| (t, None))
                .collect();
            for edge in &edges {
                ends.extend(
                    device
                        .intersect_line(*edge)
                        .iter()
                        .map(|crossing| (crossing.segment_t, Some(edge.eval(crossing.line_t)))),
                );
            }
            let length = phase.map_or(0.0, |_| segment.arclen(DASH_ACCURACY));
            for (t, crossing) in ends {
                match crossing {
                    Some(point) => points.push(point),
                    None => {
                        let point = device.eval(t);
                        points.extend(Some(point).filter(|point| reached.contains(*point)));
                        if let Some(cull) = cull_rect
                            && point.x < cull.x0
                        {
                            let moved = Point::new(cull.x0, point.y.clamp(cull.y0, cull.y1));
                            points.extend(Some(moved).filter(|point| reached.contains(*point)));
                        }
                    }
                }
                let (Some(dashes), Some(phase)) = (&dashes, phase) else {
                    continue;
                };
                let at = segment.subsegment(0.0..t).arclen(DASH_ACCURACY);
                let (start, end, _) = dashes.entry_around(phase, at);
                for stop in [start, end] {
                    if (0.0..=length).contains(&stop) {
                        let point = device.eval(segment.inv_arclen(stop, DASH_ACCURACY));
                        points.extend(Some(point).filter(|point| reached.contains(*point)));
                    }
                }
            }
        }
        if cut.is_some() {
            points.extend(corners);
        }
        let mut found: [Option<Coord>; 4] = [None, None, None, None];
        for (side, found) in found.iter_mut().enumerate() {
            // The coordinate on the side, smaller the further it reaches, and
            // the one across it.
            let along = |point: &Point| match side {
                0 => (point.x, point.y),
                1 => (point.y, point.x),
                2 => (-point.x, point.y),
                _ => (-point.y, point.x),
            };
            let mut near: Vec<(f64, f64)> = points.iter().map(along).collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut regions: Vec<Bounds> = Vec::new();
            for (value, across) in near {
                // A tile holds its point less a stroke's reach and the flattening tolerance.
                let closest = value - reach - 2.0 * TILE as f64;
                if found.as_ref().is_some_and(|found| closest > found.to_f64()) {
                    break;
                }
                let point = match side {
                    0 => (value, across),
                    1 => (across, value),
                    2 => (-value, across),
                    _ => (across, -value),
                };
                let holds = |region: &Bounds| {
                    region.x0.to_f64() <= point.0
                        && point.0 < region.x1.to_f64()
                        && region.y0.to_f64() <= point.1
                        && point.1 < region.y1.to_f64()
                };
                if regions.iter().any(holds) {
                    continue;
                }
                let origin = far_frame(side, value - reach, across, &limits);
                let region = Bounds::new(
                    &origin.0 + MARGIN,
                    &origin.1 + MARGIN,
                    &origin.0 + (FRAME - MARGIN),
                    &origin.1 + (FRAME - MARGIN),
                )
                .intersect(area);
                regions.push(region.clone());
                let Some(tiles) = Tiles::of_bounds(&region, &origin) else {
                    continue;
                };
                self.flatten(item, &origin, cull);
                let cut = cut.and_then(|cut| Tiles::of_bounds(cut, &origin));
                if let Some(tiles) = frame_tiles(&self.lines, tiles, cut, item.fill_rule()) {
                    let bounds = tiles.to_bounds(&origin);
                    let value = match side {
                        0 => bounds.x0,
                        1 => bounds.y0,
                        2 => -bounds.x1,
                        _ => -bounds.y1,
                    };
                    *found = Some(found.take().map_or(value.clone(), |found| found.min(value)));
                }
            }
        }
        match found {
            [Some(x0), Some(y0), Some(x1), Some(y1)] => Bounds::new(x0, y0, -x1, -y1),
            [None, None, None, None] => Bounds::EMPTY,
            // Reached only where frames at different origins round a dash
            // differently, so that the frame around the point reaching
            // furthest draws nothing; the side then takes the bounds of the
            // area.
            [x0, y0, x1, y1] => Bounds::new(
                x0.unwrap_or_else(|| area.x0.clone()),
                y0.unwrap_or_else(|| area.y0.clone()),
                x1.map_or_else(|| area.x1.clone(), |x1| -x1),
                y1.map_or_else(|| area.y1.clone(), |y1| -y1),
            ),
        }
    }

    /// Return the bounds of the strips `item` draws in `area` under `clips`,
    /// of which some are clip paths of any shape, and which cover the last
    /// of `covers`, see [`Self::extent`].
    ///
    /// For each side in turn, the parts of `area` are taken from the one
    /// reaching furthest there, until none left reaches further than what is
    /// found. A part shrinks to where `item` and each clip cover some tile,
    /// which is exact for each of them, until it no longer changes; a part
    /// of at most [`DRAWN`] pixels a side is then drawn under the clips, and
    /// a larger one is halved. Where the item and the clips cover tiles
    /// together along a long run, the parts reaching furthest are found by
    /// halving it. Where their edges run within a tile or two of each other
    /// along a run without covering a tile together, which only drawing
    /// tells, every part along it that reaches further than what is found is
    /// drawn, and a part drawn for one side can be drawn again for the next.
    fn search(
        &mut self,
        item: &Item<'_>,
        covers: &[Cover],
        clips: &[&Clip],
        area: Bounds,
    ) -> Bounds {
        let cull = self.cull(covers.last().unwrap(), &area);
        let mut found = Bounds::EMPTY;
        for side in 0..4 {
            // How far a part reaches on the side, larger the further.
            let reach = |part: &Bounds| match side {
                0 => -&part.x0,
                1 => -&part.y0,
                2 => part.x1.clone(),
                _ => part.y1.clone(),
            };
            let beyond =
                |part: &Bounds, found: &Bounds| found.is_empty() || reach(part) > reach(found);
            let mut queue = BinaryHeap::from([Part {
                reach: reach(&area),
                halvings: 0,
                bounds: area.clone(),
            }]);
            // Whether every part was drawn or found to hold no tiles drawn.
            let mut complete = true;
            while let Some(Part {
                halvings, bounds, ..
            }) = queue.pop()
            {
                if !beyond(&bounds, &found) {
                    complete = false;
                    break;
                }
                let part = self.shrink(item, covers, clips, cull.as_ref(), bounds);
                if part.is_empty() || found.contains(&part) {
                    continue;
                }
                if !beyond(&part, &found) {
                    complete = false;
                    continue;
                }
                if drawn_side(&part) <= DRAWN {
                    found = found.union(&self.draw(item, covers, clips, &part));
                    continue;
                }
                for half in halves(&part) {
                    queue.push(Part {
                        reach: reach(&half),
                        halvings: halvings + 1,
                        bounds: half,
                    });
                }
            }
            if complete {
                break;
            }
        }
        found
    }

    /// Return `area` shrunk to where `item`, flattened with the culling of
    /// clips whose strips lie in `cull`, and each of `clips`, which cover the
    /// last of `covers`, cover some tile, until it no longer changes.
    fn shrink(
        &mut self,
        item: &Item<'_>,
        covers: &[Cover],
        clips: &[&Clip],
        cull: Option<&Bounds>,
        mut area: Bounds,
    ) -> Bounds {
        loop {
            let before = area.clone();
            area = self.covered(item, Some(&area), cull);
            for (clip, cover) in clips.iter().zip(covers) {
                if area.is_empty() {
                    return area;
                }
                let cull = self.cull(cover, &area);
                area = self.covered(&Item::of_clip(clip), Some(&area), cull.as_ref());
            }
            if area.is_empty() || area == before {
                return area;
            }
        }
    }

    /// Return the bounds of the strips `item` draws in `area`, which a frame
    /// holds, under `clips`, which cover the last of `covers`, by drawing
    /// them as the recorder draws them, each culled to the strips of the
    /// clips under it, which a generator cut to `area` would not hold: in
    /// the recorder's viewport, in a window of it, and elsewhere in a frame.
    fn draw(&self, item: &Item<'_>, covers: &[Cover], clips: &[&Clip], area: &Bounds) -> Bounds {
        // The origin of the generator's viewport, and that of its strips.
        let (origin, strips_origin, mut generator, frame) = match self.home_holding(area) {
            Some(home) => {
                let window = home.window(area);
                let strips_origin = (
                    &home.origin.0 + i64::from(window.x0),
                    &home.origin.1 + i64::from(window.y0),
                );
                let generator =
                    StripGenerator::new_window(home.size.0, home.size.1, window, self.level);
                (home.origin.clone(), strips_origin, generator, None)
            }
            None => {
                // A frame around `area` with a margin on every side, so that
                // the strip generator is no larger: the tiler puts lines past
                // its right edge in its last column.
                let snapped = area.snap_to_tiles();
                let origin = (&snapped.x0 - MARGIN, &snapped.y0 - MARGIN);
                let side = |end: &Coord, start: &Coord| {
                    u16::try_from((end - start).clamped_i64() + MARGIN).unwrap()
                };
                let (width, height) = (side(&area.x1, &origin.0), side(&area.y1, &origin.1));
                let generator = StripGenerator::new(width, height, self.level);
                let frame = Rect::new(0.0, 0.0, width.into(), height.into());
                (origin.clone(), origin, generator, Some(frame))
            }
        };
        let to_frame = Affine::translate((-origin.0.to_f64(), -origin.1.to_f64()));
        // `generate_filled_rect_fast` cuts a rectangle under a clip path only to
        // the clip's bounds, which lie in the recorder's viewport but reach past
        // a frame; past the frame, the clip covers nothing.
        let in_frame = |rect: Rect| frame.map_or(rect, |frame| rect.intersect(frame));
        let cull = |cover: &Cover| frame_cull(self.cull(cover, area).as_ref(), &origin);
        let mut clip: Option<(StripStorage, ClipShape)> = None;
        for (pushed, cover) in clips.iter().zip(covers) {
            let mut storage = StripStorage::default();
            let shape = match pushed {
                Clip::Rect(rect) => generator.generate_filled_rect_fast(
                    &in_frame(to_frame.transform_rect_bbox(*rect)),
                    &mut storage,
                    clip_ref(&clip, cull(cover)),
                ),
                Clip::Path {
                    path,
                    fill_rule,
                    transform,
                    aliasing_threshold,
                } => {
                    generator.generate_filled_path(
                        path,
                        *fill_rule,
                        to_frame * *transform,
                        *aliasing_threshold,
                        &mut storage,
                        clip_ref(&clip, cull(cover)).map(|clip| clip.path),
                    );
                    ClipShape::Path
                }
            };
            clip = Some((storage, shape));
        }
        let existing = clip_ref(&clip, cull(covers.last().unwrap()));
        let mut storage = StripStorage::default();
        match *item {
            Item::Rect(..) => {
                generator.generate_filled_rect_fast(
                    &in_frame(item.frame_rect(to_frame).unwrap()),
                    &mut storage,
                    existing,
                );
            }
            Item::Fill {
                path,
                fill_rule,
                transform,
                aliasing_threshold,
            } => generator.generate_filled_path(
                path,
                fill_rule,
                to_frame * transform,
                aliasing_threshold,
                &mut storage,
                existing.map(|clip| clip.path),
            ),
            Item::Stroke {
                path,
                stroke,
                transform,
                aliasing_threshold,
            } => generator.generate_stroked_path(
                path,
                stroke,
                to_frame * transform,
                aliasing_threshold,
                &mut storage,
                existing.map(|clip| clip.path),
            ),
        }
        let Some(region) = Tiles::of_bounds(area, &strips_origin) else {
            return Bounds::EMPTY;
        };
        let tile = |value: i64| u16::try_from(value).unwrap();
        let region = RectU16::new(
            tile(region.x0),
            tile(region.y0),
            tile(region.x1 + 1),
            tile(region.y1 + 1),
        );
        let mut tiles: Option<Tiles> = None;
        let add = |tiles: &mut Option<Tiles>, segment: StripFillSegment| {
            let rect = segment.tile_rect();
            let segment = Tiles::new(
                i64::from(rect.x0),
                i64::from(rect.y0),
                i64::from(rect.x1) - 1,
                i64::from(rect.y1) - 1,
            );
            *tiles = Tiles::union(*tiles, segment);
        };
        visit_strip_fill_segments(
            &storage.strips,
            region,
            &mut tiles,
            |tiles, segment| add(tiles, *segment),
            add,
        );
        tiles.map_or(Bounds::EMPTY, |tiles| tiles.to_bounds(&strips_origin))
    }
}

/// Return the clip whose strips are in `clip`, culling to `bbox`.
fn clip_ref(clip: &Option<(StripStorage, ClipShape)>, bbox: RectU16) -> Option<ClipRef<'_>> {
    clip.as_ref().map(|(storage, shape)| ClipRef {
        path: PathDataRef {
            strips: &storage.strips,
            alphas: &storage.alphas,
            bbox,
            opaque_bbox: None,
        },
        shape: *shape,
    })
}

/// Return the parts of `bounds` outside `hole`: the rows above and below
/// it, and those beside it.
fn outside_of(bounds: &Bounds, hole: &Bounds) -> Vec<Bounds> {
    let hole = hole.intersect(bounds);
    if hole.is_empty() {
        return vec![bounds.clone()];
    }
    [
        Bounds::new(
            bounds.x0.clone(),
            bounds.y0.clone(),
            bounds.x1.clone(),
            hole.y0.clone(),
        ),
        Bounds::new(
            bounds.x0.clone(),
            hole.y1.clone(),
            bounds.x1.clone(),
            bounds.y1.clone(),
        ),
        Bounds::new(
            bounds.x0.clone(),
            hole.y0.clone(),
            hole.x0.clone(),
            hole.y1.clone(),
        ),
        Bounds::new(
            hole.x1.clone(),
            hole.y0.clone(),
            bounds.x1.clone(),
            hole.y1.clone(),
        ),
    ]
    .into_iter()
    .filter(|piece| !piece.is_empty())
    .collect()
}

/// Return the tiles `rect`, in the coordinates of the viewport at `origin`
/// and cut to it, covers, as `vello_common::rect::render` covers it: from the
/// tile holding its top left pixel to the one holding its right edge and its
/// bottom pixel.
fn frame_rect_tiles(rect: Rect, origin: &Position) -> Bounds {
    let finite = [rect.x0, rect.y0, rect.x1, rect.y1]
        .iter()
        .all(|value| value.is_finite());
    if !finite || rect.is_zero_area() {
        return Bounds::EMPTY;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the recorder converts coordinates to `f32`"
    )]
    let pixel = |value: f64, round: fn(f32) -> f32| Coord::from(whole(round(value as f32)));
    let (x0, y0) = (pixel(rect.x0, f32::floor), pixel(rect.y0, f32::floor));
    let (x1, y1) = (pixel(rect.x1, f32::floor), pixel(rect.y1, f32::ceil));
    Bounds::new(
        &origin.0 + tile_floor(&x0),
        &origin.1 + tile_floor(&y0),
        &origin.0 + (tile_floor(&x1) + TILE),
        &origin.1 - tile_floor(&-y1),
    )
}

/// Return `bounds`, which hold the lines of a path, with the column of
/// tiles along the left edge of `cull`, the bounds of the strips of the clips
/// it is drawn under, where what lies left of it is drawn, see `fill_clip`.
fn culled(bounds: Bounds, cull: Option<&Bounds>) -> Bounds {
    match cull {
        Some(cull) if !cull.is_empty() && bounds.x0 < cull.x0 => bounds.union(&Bounds::new(
            cull.x0.clone(),
            bounds.y0.clone(),
            &cull.x0 + TILE,
            bounds.y1.clone(),
        )),
        _ => bounds,
    }
}

/// A part [`Extents::search`] has yet to look at, which it takes from those
/// reaching furthest on the side it searches, and of those from the most
/// often halved: the halves of a part are then taken before the parts that
/// were waiting, so that those waiting are about one for each halving rather
/// than one for each part drawn along a run.
struct Part {
    reach: Coord,
    halvings: u32,
    bounds: Bounds,
}

impl Ord for Part {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.reach, self.halvings).cmp(&(&other.reach, other.halvings))
    }
}

impl PartialOrd for Part {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Part {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Part {}

/// The longest side of the parts [`Extents::search`] draws, which bounds
/// what drawing one costs.
const DRAWN: i64 = 1024;

/// Return the longer side of `part`.
fn drawn_side(part: &Bounds) -> i64 {
    part.width().max(part.height()).clamped_i64()
}

/// Return the halves of `area` across its longer side, split on a tile.
fn halves(area: &Bounds) -> [Bounds; 2] {
    let (mut low, mut high) = (area.clone(), area.clone());
    if area.width() >= area.height() {
        let middle = tile_floor(&(&area.x0 + &area.x1).shr(1));
        low.x1 = middle.clone();
        high.x0 = middle;
    } else {
        let middle = tile_floor(&(&area.y0 + &area.y1).shr(1));
        low.y1 = middle.clone();
        high.y0 = middle;
    }
    [low, high]
}

/// Return whether a frame holds `area` inside its margin.
fn fits(area: &Bounds) -> bool {
    let side = Coord::from(FRAME - 2 * MARGIN);
    area.width() <= side && area.height() <= side
}

/// Return the culling rectangle of clips whose strips lie in `cull`, or of
/// none, in the frame at `origin`, see `StripGenerator::generate_filled_path`;
/// where `cull` reaches past the frame, it is cut to the frame, which changes
/// only what the frame's margin holds.
fn frame_cull(cull: Option<&Bounds>, origin: &Position) -> RectU16 {
    let local = |value: &Coord, origin: &Coord| {
        u16::try_from((value - origin).clamped_i64().clamp(0, FRAME)).unwrap()
    };
    match cull {
        None => {
            let side = u16::try_from(FRAME).unwrap();
            RectU16::new(0, 0, side, side)
        }
        Some(cull) => RectU16::new(
            local(&cull.x0, &origin.0),
            local(&cull.y0, &origin.1),
            local(&cull.x1, &origin.0),
            local(&cull.y1, &origin.1),
        ),
    }
}

/// Return the tiles in `region` that `lines`, flattened in a frame, cover
/// under `fill_rule`, cut to `cut`, the tiles a clip covers, if there is one.
///
/// Each line covers the tiles it crosses. Between them, a row is filled
/// only to the right of a line crossing it, so inside `cut`, the tiles that
/// reach furthest are crossed by a line or lie in the first or last column
/// of `cut`, where the fill rule fills them, see [`filled_rows`].
fn frame_tiles(
    lines: &[Line],
    region: Tiles,
    cut: Option<Tiles>,
    fill_rule: Fill,
) -> Option<Tiles> {
    let region = match cut {
        Some(cut) => region.intersect(cut)?,
        None => region,
    };
    let mut tiles = None;
    for line in lines {
        if let Some(line) = TiledLine::new(*line) {
            tiles = Tiles::union(tiles, line.within(region));
        }
    }
    if let Some(cut) = cut {
        for column in [cut.x0, cut.x1] {
            if (region.x0..=region.x1).contains(&column)
                && let Some((first, last)) =
                    filled_rows(lines, column, (region.y0, region.y1), fill_rule)
            {
                tiles = Tiles::union(tiles, Tiles::new(column, first, column, last));
            }
        }
    }
    tiles
}

/// Return the origin of the frame on `side` (left, top, right, bottom)
/// around a point whose coordinate on that side, smaller the further it
/// reaches and less a stroke's reach, is `value`, and whose coordinate across
/// it is `across`, inside `limits`: the frame's edge on that side lies a
/// margin beyond the point.
fn far_frame(side: usize, value: f64, across: f64, limits: &Rect) -> Position {
    let tile = |value: f64| tile_floor(&Coord::floor_of(value));
    let across = tile(across) - (FRAME / 2 / TILE * TILE);
    match side {
        0 => (tile(value.max(limits.x0)) - MARGIN, across),
        1 => (across, tile(value.max(limits.y0)) - MARGIN),
        2 => (
            tile((-value).min(limits.x1)) + (MARGIN + TILE - FRAME),
            across,
        ),
        _ => (
            across,
            tile((-value).min(limits.y1)) + (MARGIN + TILE - FRAME),
        ),
    }
}

/// Return whether `a` and `b` share a point.
fn overlap(a: Rect, b: Rect) -> bool {
    a.x0 <= b.x1 && b.x0 <= a.x1 && a.y0 <= b.y1 && b.y0 <= a.y1
}

/// Return whether `item` is a path that outlines an axis-aligned rectangle
/// once, under a transform that keeps its sides axis-aligned: four lines,
/// each joining the last, along one axis and then the other, back to where
/// they start. Its winding is then one inside, which either fill rule
/// fills, and its sides have equal coordinates in every frame, so it covers
/// every tile its lines cross.
fn is_axis_aligned_rect(item: &Item<'_>) -> bool {
    if !matches!(item, Item::Fill { .. }) {
        return false;
    }
    let (segments, transform) = item.segments(None);
    let [a, b, c, d, _, _] = transform.as_coeffs();
    if !(b == 0.0 && c == 0.0 || a == 0.0 && d == 0.0) {
        return false;
    }
    let lines: Vec<Segment> = segments
        .iter()
        .filter_map(|(segment, _)| match segment {
            PathSeg::Line(line) => Some(*line),
            PathSeg::Quad(_) | PathSeg::Cubic(_) => None,
        })
        .collect();
    if lines.len() != 4 || segments.len() != 4 {
        return false;
    }
    let along_x = |line: &Segment| line.p0.y == line.p1.y && line.p0.x != line.p1.x;
    let along_y = |line: &Segment| line.p0.x == line.p1.x && line.p0.y != line.p1.y;
    (0..4).all(|index| {
        let (line, next) = (&lines[index], &lines[(index + 1) % 4]);
        line.p1 == next.p0 && (along_x(line) && along_y(next) || along_y(line) && along_x(next))
    })
}

/// Return `bounds` as a rectangle in `f64`, whose sides are not finite where
/// the bounds are empty.
fn to_rect(bounds: &Bounds) -> Rect {
    if bounds.is_empty() {
        return Rect::new(0.0, 0.0, 0.0, 0.0);
    }
    Rect::new(
        bounds.x0.to_f64(),
        bounds.y0.to_f64(),
        bounds.x1.to_f64(),
        bounds.y1.to_f64(),
    )
}

#[cfg(all(test, feature = "std"))]
mod tests;
