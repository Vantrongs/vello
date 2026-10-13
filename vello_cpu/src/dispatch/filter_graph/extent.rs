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

use super::{Clip, Command, FilterGraph, Layer, Step, WINDOW_SIZE};
use crate::filter::placed::{Bounds, Coord, Position};
use crate::kurbo::{
    Affine, BezPath, Join, Line as Segment, ParamCurve, ParamCurveExtrema, PathEl, PathSeg, Point,
    Rect, Stroke, StrokeCtx,
};
use crate::peniko::Fill;
#[cfg(not(feature = "std"))]
use crate::peniko::kurbo::common::FloatFuncs as _;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::clip::{ClipRef, ClipShape, PathDataRef};
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
    Rect(Rect),
    Fill {
        path: &'a BezPath,
        fill_rule: Fill,
        transform: Affine,
    },
    Stroke {
        path: &'a BezPath,
        stroke: &'a Stroke,
        transform: Affine,
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
                ..
            } => Some(Self::Fill {
                path,
                fill_rule: *fill_rule,
                transform: *transform,
            }),
            Command::StrokePath {
                path,
                stroke,
                transform,
                ..
            } => Some(Self::Stroke {
                path,
                stroke,
                transform: *transform,
            }),
            Command::FillRect { rect, .. } => Some(Self::Rect(*rect)),
            _ => None,
        }
    }

    fn of_clip(clip: &'a Clip) -> Self {
        match clip {
            Clip::Path {
                path,
                fill_rule,
                transform,
                ..
            } => Self::Fill {
                path,
                fill_rule: *fill_rule,
                transform: *transform,
            },
            Clip::Rect(rect) => Self::Rect(*rect),
        }
    }

    /// Return the clip path of `layer`, if it has one.
    fn of_layer_clip(layer: &'a Layer) -> Option<Self> {
        layer.clip_path.as_ref().map(|path| Self::Fill {
            path,
            fill_rule: layer.fill_rule,
            transform: layer.clip_transform,
        })
    }

    fn fill_rule(&self) -> Fill {
        match self {
            Self::Fill { fill_rule, .. } => *fill_rule,
            Self::Rect(_) | Self::Stroke { .. } => Fill::NonZero,
        }
    }

    /// Return how far a stroke's outline reaches from its path in device
    /// pixels, as [`Command::bounds`] counts it, and zero for fills.
    fn reach(&self) -> f64 {
        match self {
            Self::Stroke {
                stroke, transform, ..
            } => {
                let miter = if stroke.join == Join::Miter {
                    stroke.miter_limit.max(core::f64::consts::SQRT_2)
                } else {
                    core::f64::consts::SQRT_2
                };
                let [a, b, c, d, _, _] = transform.as_coeffs();
                miter * stroke.width.abs() / 2.0 * (a.abs() + b.abs() + c.abs() + d.abs())
            }
            Self::Rect(_) | Self::Fill { .. } => 0.0,
        }
    }

    /// Return a rectangle in root target coordinates that holds the lines the
    /// item is flattened to.
    fn device_bounds(&self) -> Rect {
        let (path, transform) = match self {
            Self::Rect(rect) => return rect.abs(),
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

    /// Return the segments of the path in root target coordinates, with the
    /// lines that close each subpath of a fill, as the flattener walks them.
    fn device_segments(&self) -> Vec<PathSeg> {
        let (path, transform, closes) = match self {
            Self::Rect(_) => return Vec::new(),
            Self::Fill {
                path, transform, ..
            } => (path, transform, true),
            Self::Stroke {
                path, transform, ..
            } => (path, transform, false),
        };
        let mut segments = Vec::new();
        let mut elements = path.iter().map(|element| *transform * element);
        let Some(PathEl::MoveTo(first)) = elements.next() else {
            return segments;
        };
        let (mut start, mut last) = (first, first);
        let close = |segments: &mut Vec<PathSeg>, last: Point, start: Point| {
            if closes && last != start {
                segments.push(PathSeg::Line(Segment::new(last, start)));
            }
        };
        for element in elements {
            match element {
                PathEl::MoveTo(point) => {
                    close(&mut segments, last, start);
                    (start, last) = (point, point);
                }
                PathEl::LineTo(point) => {
                    segments.push(PathSeg::Line(Segment::new(last, point)));
                    last = point;
                }
                PathEl::QuadTo(control, point) => {
                    segments.push(PathSeg::Quad(crate::kurbo::QuadBez::new(
                        last, control, point,
                    )));
                    last = point;
                }
                PathEl::CurveTo(first_control, second_control, point) => {
                    segments.push(PathSeg::Cubic(crate::kurbo::CubicBez::new(
                        last,
                        first_control,
                        second_control,
                        point,
                    )));
                    last = point;
                }
                PathEl::ClosePath => {
                    if last != start {
                        segments.push(PathSeg::Line(Segment::new(last, start)));
                    }
                    last = start;
                }
            }
        }
        close(&mut segments, last, start);
        segments
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
    /// Clips of other shapes, whose strips lie in the bounds.
    Paths(Bounds),
}

/// Finds the bounds of the strips the commands of a filter layer draw.
pub(super) struct Extents<'a> {
    level: Level,
    /// The viewport that bounds commands drawing without bounds, see `Node`.
    viewport: &'a Bounds,
    lines: Vec<Line>,
    flatten: FlattenCtx,
    stroke: StrokeCtx,
}

impl<'a> Extents<'a> {
    pub(super) fn new(level: Level, viewport: &'a Bounds) -> Self {
        Self {
            level,
            viewport,
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

    /// Return what the clips cover once `clip` is pushed onto `clips`, which
    /// cover the last of `covers`, see `ClipContext::push_clip_rect`.
    fn push_clip(&mut self, covers: &[Cover], clips: &[&Clip], clip: &Clip) -> Cover {
        let item = Item::of_clip(clip);
        match (covers.last().unwrap(), clip) {
            (Cover::All, Clip::Rect(rect)) => Cover::Rect(self.unbounded_cut(rect.abs())),
            (Cover::Rect(clip), Clip::Rect(rect)) => Cover::Rect(rect.abs().intersect(*clip)),
            (Cover::Tiles(tiles), Clip::Rect(rect)) => {
                let rect = rect.abs().intersect(to_rect(tiles));
                Cover::Tiles(self.rect_tiles(rect).intersect(tiles))
            }
            // A rectangle covers every tile its lines cross.
            (Cover::All | Cover::Rect(_) | Cover::Tiles(_), Clip::Path { .. })
                if is_axis_aligned_rect(&item) =>
            {
                Cover::Tiles(self.extent(&item, covers, clips))
            }
            _ => Cover::Paths(self.extent(&item, covers, clips)),
        }
    }

    /// Return the bounds of the strips of the clips that cover `cover`,
    /// which the recorder culls what it flattens under them to, see
    /// `StripGenerator::generate_filled_path`, or `None` without clips.
    fn clip_bounds(&self, cover: &Cover) -> Option<Bounds> {
        match cover {
            Cover::All => None,
            Cover::Rect(rect) => Some(self.rect_tiles(*rect)),
            Cover::Tiles(bounds) | Cover::Paths(bounds) => Some(bounds.clone()),
        }
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
    /// each next that of one more of `clips`.
    fn extent(&mut self, item: &Item<'_>, covers: &[Cover], clips: &[&Clip]) -> Bounds {
        let cover = covers.last().unwrap();
        let cull = self.clip_bounds(cover);
        match (item, cover) {
            (Item::Rect(rect), Cover::All) => self.rect_tiles(self.unbounded_cut(rect.abs())),
            (Item::Rect(rect), Cover::Rect(clip)) => self.rect_tiles(rect.abs().intersect(*clip)),
            (Item::Rect(rect), Cover::Tiles(tiles)) => self
                .rect_tiles(rect.abs().intersect(to_rect(tiles)))
                .intersect(tiles),
            (_, Cover::All) => self.path(item, None, None),
            (_, Cover::Rect(_) | Cover::Tiles(_)) => self.path(item, cull.as_ref(), cull.as_ref()),
            (_, Cover::Paths(bounds)) => {
                let area = self.covered(item, Some(bounds), cull.as_ref());
                let mut found = Bounds::EMPTY;
                self.search(item, covers, clips, area, &mut found);
                found
            }
        }
    }

    /// Return the bounds of the strips `item` draws inside `cut`, a block of
    /// tiles a clip covers, or anywhere if there is none, flattened with the
    /// culling of clips whose strips lie in `cull`.
    fn covered(&mut self, item: &Item<'_>, cut: Option<&Bounds>, cull: Option<&Bounds>) -> Bounds {
        match item {
            Item::Rect(rect) => {
                let tiles = self.rect_tiles(self.unbounded_cut(rect.abs()));
                cut.map_or(tiles.clone(), |cut| tiles.intersect(cut))
            }
            _ => self.path(item, cut, cull),
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
        let area = cut.map_or(bounds.clone(), |cut| bounds.intersect(cut));
        if area.is_empty() {
            return Bounds::EMPTY;
        }
        if !fits(&area) {
            return self.far_path(item, &area, cut, cull);
        }
        let origin = self.frame_at(&area);
        self.flatten(item, &origin, cull);
        let region = Tiles::of_bounds(&area, &origin).unwrap();
        let cut = cut.and_then(|cut| Tiles::of_bounds(cut, &origin));
        frame_tiles(&self.lines, region, cut, item.fill_rule())
            .map_or(Bounds::EMPTY, |tiles| tiles.to_bounds(&origin))
    }

    /// Return the origin of a frame holding `area` inside its margin: that of
    /// the viewport if it does, where the recorder flattens, and otherwise
    /// one at the top left of `area`.
    fn frame_at(&self, area: &Bounds) -> Position {
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
            Item::Rect(_) => unreachable!("rectangles are not flattened"),
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
    /// segments do, where they cross the edges of `area`, and the corners of
    /// `cut`, which a fill can cover. From the furthest, the path is flattened
    /// in a frame around each point not yet in one, with that side of the
    /// frame beyond the point by a stroke's reach, until no point is left
    /// that reaches further than the tiles found.
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
        let edges = [
            Segment::new(corners[0], corners[1]),
            Segment::new(corners[2], corners[3]),
            Segment::new(corners[0], corners[2]),
            Segment::new(corners[1], corners[3]),
        ];
        let mut points: Vec<Point> = Vec::new();
        for segment in item.device_segments() {
            if !overlap(segment.bounding_box(), reached) {
                continue;
            }
            let ends = [0.0, 1.0].into_iter().chain(segment.extrema());
            points.extend(
                ends.map(|t| segment.eval(t))
                    .filter(|point| reached.contains(*point)),
            );
            for edge in &edges {
                points.extend(
                    segment
                        .intersect_line(*edge)
                        .iter()
                        .map(|crossing| edge.eval(crossing.line_t)),
                );
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
            // Not reached by geometry the flattener draws whole near the
            // points reaching furthest; a dashed stroke with gaps longer than
            // a frame there can. The side then takes the bounds of the area.
            [x0, y0, x1, y1] => Bounds::new(
                x0.unwrap_or_else(|| area.x0.clone()),
                y0.unwrap_or_else(|| area.y0.clone()),
                x1.map_or_else(|| area.x1.clone(), |x1| -x1),
                y1.map_or_else(|| area.y1.clone(), |y1| -y1),
            ),
        }
    }

    /// Add to `found` the bounds of the strips `item` draws in `area` under
    /// `clips`, of which some are clip paths of any shape, and which cover
    /// the last of `covers`, see [`Self::extent`].
    ///
    /// `area` shrinks to where `item` and each clip cover some tile, which
    /// is exact for each of them, until it no longer changes. Inside a
    /// frame, the item is drawn under the clips; a larger area is halved.
    /// Where the edges of the item and a clip run close together over many
    /// frames, every frame along them is drawn.
    fn search(
        &mut self,
        item: &Item<'_>,
        covers: &[Cover],
        clips: &[&Clip],
        mut area: Bounds,
        found: &mut Bounds,
    ) {
        let cull = self.clip_bounds(covers.last().unwrap());
        loop {
            let before = area.clone();
            area = self.covered(item, Some(&area), cull.as_ref());
            for (clip, cover) in clips.iter().zip(covers) {
                if area.is_empty() {
                    break;
                }
                let cull = self.clip_bounds(cover);
                area = self.covered(&Item::of_clip(clip), Some(&area), cull.as_ref());
            }
            if area.is_empty() || area == before {
                break;
            }
        }
        if area.is_empty() || found.contains(&area) {
            return;
        }
        if fits(&area) {
            *found = found.union(&self.draw(item, covers, clips, &area));
            return;
        }
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
        self.search(item, covers, clips, low, found);
        self.search(item, covers, clips, high, found);
    }

    /// Return the bounds of the strips `item` draws in `area`, which a frame
    /// holds, under `clips`, which cover the last of `covers`, by drawing
    /// them in that frame as the recorder draws them: each culled to the
    /// strips of the clips under it in the whole target, which a window cut
    /// to `area` would not hold.
    fn draw(&self, item: &Item<'_>, covers: &[Cover], clips: &[&Clip], area: &Bounds) -> Bounds {
        let origin = self.frame_at(area);
        let to_frame = Affine::translate((-origin.0.to_f64(), -origin.1.to_f64()));
        let side = |end: &Coord, start: &Coord| {
            u16::try_from(((end - start).clamped_i64() + TILE - 1) / TILE * TILE).unwrap()
        };
        let mut generator = StripGenerator::new(
            side(&area.x1, &origin.0),
            side(&area.y1, &origin.1),
            self.level,
        );
        let cull = |cover: &Cover| frame_cull(self.clip_bounds(cover).as_ref(), &origin);
        let mut clip: Option<(StripStorage, ClipShape)> = None;
        for (pushed, cover) in clips.iter().zip(covers) {
            let mut storage = StripStorage::default();
            let shape = match pushed {
                Clip::Rect(rect) => generator.generate_filled_rect_fast(
                    &to_frame.transform_rect_bbox(*rect),
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
            Item::Rect(rect) => {
                generator.generate_filled_rect_fast(
                    &to_frame.transform_rect_bbox(rect),
                    &mut storage,
                    existing,
                );
            }
            Item::Fill {
                path,
                fill_rule,
                transform,
            } => generator.generate_filled_path(
                path,
                fill_rule,
                to_frame * transform,
                None,
                &mut storage,
                existing.map(|clip| clip.path),
            ),
            Item::Stroke {
                path,
                stroke,
                transform,
            } => generator.generate_stroked_path(
                path,
                stroke,
                to_frame * transform,
                None,
                &mut storage,
                existing.map(|clip| clip.path),
            ),
        }
        let Some(region) = Tiles::of_bounds(area, &origin) else {
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
        tiles.map_or(Bounds::EMPTY, |tiles| tiles.to_bounds(&origin))
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

/// Return whether `item` is a path whose device lines outline an
/// axis-aligned rectangle, which covers every tile its lines cross.
fn is_axis_aligned_rect(item: &Item<'_>) -> bool {
    let segments = item.device_segments();
    let mut corners: Vec<Point> = Vec::with_capacity(4);
    for segment in &segments {
        let PathSeg::Line(line) = segment else {
            return false;
        };
        if line.p0.x != line.p1.x && line.p0.y != line.p1.y {
            return false;
        }
        if !corners.contains(&line.p0) {
            corners.push(line.p0);
        }
        if corners.len() > 4 {
            return false;
        }
    }
    let xs = corners
        .iter()
        .filter(|corner| corner.x == corners[0].x)
        .count();
    let ys = corners
        .iter()
        .filter(|corner| corner.y == corners[0].y)
        .count();
    corners.len() == 4 && segments.len() == 4 && xs == 2 && ys == 2
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
