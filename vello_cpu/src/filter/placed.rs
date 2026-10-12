// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter pixmaps placed in a source space that can exceed the `u16` domain.

use crate::filter::pixmap::FilterPixmap;
use crate::kurbo::Rect;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::geometry::{PaddingU64, RectU16};
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::tile::Tile;

/// The bound of the coordinates of [`Bounds`] in each pixel grid. Lengths and
/// sums of two coordinates fit `i64`, and areas fit `u128`.
///
/// It reaches far past the precision of the `f64` geometry drawn into the
/// grid, so nothing is placed beyond it.
const LIMIT: i64 = 1 << 61;

/// Return the width of the pixels of the grid `level` steps of two coarser
/// than the root target's, in root target pixels.
pub(crate) fn pixel_size(level: u32) -> f64 {
    (0..level).fold(1.0, |size, _| size * 2.0)
}

/// A whole number of pixels, `high * 2^64 + low` with `low` in `0..2^64`, so
/// that the offsets of any number of filters, each up to `f32::MAX`, add up
/// exactly and opposite ones cancel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Shift {
    high: i128,
    low: i128,
}

impl Shift {
    /// Return the whole number of pixels `value`.
    pub(crate) fn of(value: f32) -> Self {
        // An `f32` this large is a multiple of `2^64`, and `i128` holds the rest exactly.
        if value.abs() >= (1_u128 << 126) as f32 {
            Self {
                high: (value / (1_u128 << 64) as f32) as i128,
                low: 0,
            }
        } else {
            Self::normalized(0, value as i128)
        }
    }

    fn normalized(high: i128, low: i128) -> Self {
        Self {
            high: high + (low >> 64),
            low: low & i128::from(u64::MAX),
        }
    }

    pub(crate) fn add(self, other: Self) -> Self {
        Self::normalized(self.high + other.high, self.low + other.low)
    }

    /// Return the number, saturated at `±i64::MAX`, which lie beyond the
    /// domain of every grid and can be negated.
    pub(crate) fn saturated(self) -> i64 {
        let value = self
            .high
            .checked_mul(1 << 64)
            .and_then(|high| high.checked_add(self.low))
            .unwrap_or(if self.high < 0 { i128::MIN } else { i128::MAX });
        value.clamp((-i64::MAX).into(), i64::MAX.into()) as i64
    }
}

/// A rectangle in a pixel grid, with coordinates in `-LIMIT..=LIMIT`: whatever
/// lies outside that domain is cut off. Empty rectangles are normalized to
/// [`Bounds::EMPTY`], so their width and height are never negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bounds {
    pub(crate) x0: i64,
    pub(crate) y0: i64,
    pub(crate) x1: i64,
    pub(crate) y1: i64,
}

impl Bounds {
    pub(crate) const EMPTY: Self = Self {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    /// The whole domain of a grid.
    pub(crate) const DOMAIN: Self = Self {
        x0: -LIMIT,
        y0: -LIMIT,
        x1: LIMIT,
        y1: LIMIT,
    };

    pub(crate) fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Self {
        Self::clamped(x0.into(), y0.into(), x1.into(), y1.into())
    }

    /// Return the part of the rectangle inside the domain.
    fn clamped(x0: i128, y0: i128, x1: i128, y1: i128) -> Self {
        let clamp = |value: i128| value.clamp(-i128::from(LIMIT), i128::from(LIMIT)) as i64;
        let (x0, y0, x1, y1) = (clamp(x0), clamp(y0), clamp(x1), clamp(y1));
        if x0 >= x1 || y0 >= y1 {
            return Self::EMPTY;
        }
        Self { x0, y0, x1, y1 }
    }

    /// Return the bounds of the pixels that `rect` touches in the grid of
    /// pixels `1 / scale` wide; all of the domain if `rect` is not finite.
    pub(crate) fn covering(rect: Rect, scale: f64) -> Self {
        let rect = rect.scale_from_origin(scale);
        if !(rect.x0.is_finite()
            && rect.y0.is_finite()
            && rect.x1.is_finite()
            && rect.y1.is_finite())
        {
            return Self::DOMAIN;
        }
        // Converting saturates, which the domain then clamps.
        Self::new(
            rect.x0.floor() as i64,
            rect.y0.floor() as i64,
            rect.x1.ceil() as i64,
            rect.y1.ceil() as i64,
        )
    }

    /// Return the bounds of a viewport of the given size.
    pub(crate) fn of_viewport(width: u16, height: u16) -> Self {
        Self::new(0, 0, i64::from(width), i64::from(height))
    }

    /// Return the bounds of `rect`, which is relative to `origin`.
    pub(crate) fn of_rect(rect: RectU16, origin: (i64, i64)) -> Self {
        Self::new(
            i64::from(rect.x0) + origin.0,
            i64::from(rect.y0) + origin.1,
            i64::from(rect.x1) + origin.0,
            i64::from(rect.y1) + origin.1,
        )
    }

    pub(crate) fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    pub(crate) fn width(self) -> i64 {
        self.x1 - self.x0
    }

    pub(crate) fn height(self) -> i64 {
        self.y1 - self.y0
    }

    pub(crate) fn area(self) -> u128 {
        self.width() as u128 * self.height() as u128
    }

    pub(crate) fn translate(self, (dx, dy): (i64, i64)) -> Self {
        if self.is_empty() {
            return self;
        }
        let (dx, dy) = (i128::from(dx), i128::from(dy));
        Self::clamped(
            i128::from(self.x0) + dx,
            i128::from(self.y0) + dy,
            i128::from(self.x1) + dx,
            i128::from(self.y1) + dy,
        )
    }

    pub(crate) fn expand(self, padding: PaddingU64) -> Self {
        if self.is_empty() {
            return self;
        }
        Self::clamped(
            i128::from(self.x0) - i128::from(padding.left),
            i128::from(self.y0) - i128::from(padding.top),
            i128::from(self.x1) + i128::from(padding.right),
            i128::from(self.y1) + i128::from(padding.bottom),
        )
    }

    pub(crate) fn intersect(self, other: Self) -> Self {
        Self::new(
            self.x0.max(other.x0),
            self.y0.max(other.y0),
            self.x1.min(other.x1),
            self.y1.min(other.y1),
        )
    }

    /// Return the smallest bounds containing both.
    pub(crate) fn union(self, other: Self) -> Self {
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

    pub(crate) fn snap_to_tiles(self) -> Self {
        if self.is_empty() {
            return self;
        }
        let (width, height) = (i64::from(Tile::WIDTH), i64::from(Tile::HEIGHT));
        Self::new(
            self.x0.div_euclid(width) * width,
            self.y0.div_euclid(height) * height,
            (self.x1 + width - 1).div_euclid(width) * width,
            (self.y1 + height - 1).div_euclid(height) * height,
        )
    }

    /// Return the bounds of the pixels `levels` times coarser by a factor of
    /// two that cover these bounds, on a grid aligned with this one at zero.
    pub(crate) fn coarsen(self, levels: u32) -> Self {
        if self.is_empty() {
            return self;
        }
        // Past 64 levels, every coordinate of the domain falls into the
        // coarse pixels next to zero.
        let step = 1_i128 << levels.min(64);
        let start = |value: i64| i128::from(value).div_euclid(step);
        let end = |value: i64| (i128::from(value) + step - 1).div_euclid(step);
        Self::clamped(start(self.x0), start(self.y0), end(self.x1), end(self.y1))
    }

    /// Return the bounds of these pixels on a grid `levels` times finer by a
    /// factor of two, as far as that grid's domain reaches.
    pub(crate) fn refine(self, levels: u32) -> Self {
        // Past 64 levels, every coordinate but zero leaves the domain.
        let scale = |value: i64| i128::from(value) << levels.min(64);
        Self::clamped(
            scale(self.x0),
            scale(self.y0),
            scale(self.x1),
            scale(self.y1),
        )
    }

    /// Convert bounds relative to `origin` into a `u16` rectangle, clipping the
    /// exclusive edge to the `u16` domain.
    pub(crate) fn to_rect_u16(self, origin: (i64, i64)) -> RectU16 {
        if self.is_empty() {
            return RectU16::ZERO;
        }
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

/// Bounds indexed by where they lie, so that the members touching a query
/// are found without visiting the others.
///
/// A member whose longer side is at most `2^class` lies in the cell of that
/// size holding its top left corner, so the members of a class that touch a
/// query lie in the cells from the one before the query's start to the one at
/// its end.
#[derive(Clone, Debug, Default)]
pub(crate) struct Index {
    /// Each member by class, cell row, cell column and id.
    members: BTreeMap<(u32, i64, i64, usize), Bounds>,
    /// The classes that may hold members, as bits.
    classes: u64,
}

impl Index {
    /// Index `bounds` by their position in the sequence.
    pub(crate) fn new(bounds: impl IntoIterator<Item = Bounds>) -> Self {
        let mut index = Self::default();
        for (id, bounds) in bounds.into_iter().enumerate() {
            index.insert(id, bounds);
        }
        index
    }

    fn key(id: usize, bounds: Bounds) -> (u32, i64, i64, usize) {
        let side = bounds.width().max(bounds.height()).max(1).unsigned_abs();
        let class = u64::BITS - (side - 1).leading_zeros();
        (class, bounds.y0 >> class, bounds.x0 >> class, id)
    }

    /// Add `bounds` with `id`, unless they are empty.
    pub(crate) fn insert(&mut self, id: usize, bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        let key = Self::key(id, bounds);
        self.classes |= 1 << key.0;
        self.members.insert(key, bounds);
    }

    /// Remove the member `bounds` with `id`.
    pub(crate) fn remove(&mut self, id: usize, bounds: Bounds) {
        self.members.remove(&Self::key(id, bounds));
    }

    /// Return the members, in no particular order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = Bounds> + '_ {
        self.members.values().copied()
    }

    /// Return the ids and bounds of the members that overlap `query` or share
    /// an edge with it, in no particular order.
    pub(crate) fn touching(&self, query: Bounds) -> impl Iterator<Item = (usize, Bounds)> + '_ {
        let classes = (0..u64::BITS).filter(|class| self.classes & (1 << class) != 0);
        let candidates = classes.flat_map(move |class| {
            let cell = |value: i64| value >> class;
            let first = |value: i64| ((i128::from(value) - (1_i128 << class)) >> class) as i64;
            let (columns, rows) = (
                first(query.x0)..=cell(query.x1),
                first(query.y0)..=cell(query.y1),
            );
            // A tall query scans the members of its rows; a short one only
            // the cells it can touch.
            let short = rows.end() - rows.start() < 4;
            let bands = if short {
                (*rows.start()..=*rows.end())
                    .map(|row| {
                        (class, row, *columns.start(), 0)..=(class, row, *columns.end(), usize::MAX)
                    })
                    .collect::<Vec<_>>()
            } else {
                alloc::vec![
                    (class, *rows.start(), i64::MIN, 0)
                        ..=(class, *rows.end(), i64::MAX, usize::MAX)
                ]
            };
            bands.into_iter().flat_map(move |band| {
                let columns = columns.clone();
                self.members
                    .range(band)
                    .filter(move |((_, _, column, _), _)| columns.contains(column))
            })
        });
        candidates
            .filter(move |(_, bounds)| {
                bounds.x0 <= query.x1
                    && query.x0 <= bounds.x1
                    && bounds.y0 <= query.y1
                    && query.y0 <= bounds.y1
            })
            .map(|(&(_, _, _, id), &bounds)| (id, bounds))
    }
}

/// Regions to render. A region that is added is merged with a member whose
/// bounds together with it cover no more area than the two apart, so shared
/// pixels are rendered once and no pixel neither needs is rendered.
#[derive(Clone, Debug, Default)]
pub(crate) struct Regions {
    regions: Vec<Bounds>,
    index: Index,
}

impl Regions {
    /// Return the regions with only `bounds`.
    pub(crate) fn of(bounds: Bounds) -> Self {
        let mut regions = Self::default();
        regions.insert(bounds);
        regions
    }

    pub(crate) fn insert(&mut self, mut bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        // Regions merge only if they touch: a gap along either axis makes
        // their bounds together larger than the two apart.
        while let Some(index) = self
            .index
            .touching(bounds)
            .filter(|(_, region)| region.union(bounds).area() <= region.area() + bounds.area())
            .map(|(index, _)| index)
            .min()
        {
            bounds = bounds.union(self.swap_remove(index));
        }
        self.index.insert(self.regions.len(), bounds);
        self.regions.push(bounds);
    }

    fn swap_remove(&mut self, index: usize) -> Bounds {
        let last = self.regions.len() - 1;
        self.index.remove(index, self.regions[index]);
        if index != last {
            self.index.remove(last, self.regions[last]);
            self.index.insert(index, self.regions[last]);
        }
        self.regions.swap_remove(index)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Bounds> + '_ {
        self.regions.iter().copied()
    }

    /// Return the regions that overlap `bounds` or share an edge with it, in
    /// the order of [`Self::iter`].
    pub(crate) fn touching(&self, bounds: Bounds) -> impl Iterator<Item = Bounds> + '_ {
        let mut indices: Vec<usize> = self
            .index
            .touching(bounds)
            .map(|(index, _)| index)
            .collect();
        indices.sort_unstable();
        indices.into_iter().map(|index| self.regions[index])
    }
}

/// Placed pixmaps, indexed by their bounds.
#[derive(Debug, Default)]
pub(crate) struct Pieces {
    pieces: Vec<PlacedPixmap>,
    index: Index,
}

impl Pieces {
    pub(crate) fn new(pieces: Vec<PlacedPixmap>) -> Self {
        Self {
            index: Index::new(pieces.iter().map(|piece| piece.bounds)),
            pieces,
        }
    }

    /// Return the bounds of the pieces.
    pub(crate) fn index(&self) -> &Index {
        &self.index
    }

    /// Return a copy of the pixels of the pieces inside `area`, transparent
    /// where they do not cover it.
    pub(crate) fn crop(&self, area: Bounds) -> PlacedPixmap {
        let mut crop = PlacedPixmap::new(area);
        let mut touching: Vec<usize> = self.index.touching(area).map(|(index, _)| index).collect();
        // The order of the pieces decides the pixels where they overlap.
        touching.sort_unstable();
        for index in touching {
            crop.copy_from(&self.pieces[index], area);
        }
        crop
    }
}

/// A filter pixmap covering `bounds`.
#[derive(Debug)]
pub(crate) struct PlacedPixmap {
    pub(crate) pixmap: FilterPixmap,
    pub(crate) bounds: Bounds,
}

impl PlacedPixmap {
    /// Create a transparent pixmap covering `bounds`.
    pub(crate) fn new(bounds: Bounds) -> Self {
        let size = |length: i64| u32::try_from(length).expect("filter pixmaps fit the u32 domain");
        Self {
            pixmap: FilterPixmap::new(size(bounds.width()), size(bounds.height())),
            bounds,
        }
    }

    pub(crate) fn translate(self, offset: (i64, i64)) -> Self {
        Self {
            bounds: self.bounds.translate(offset),
            ..self
        }
    }

    /// Return a copy of the pixels inside `area`, transparent where this pixmap
    /// does not cover it.
    pub(crate) fn crop(&self, area: Bounds) -> Self {
        let mut crop = Self::new(area);
        crop.copy_from(self, area);
        crop
    }

    /// Copy the pixels of `source` inside `area`.
    pub(crate) fn copy_from(&mut self, source: &Self, area: Bounds) {
        self.combine(source, area, |target, source| {
            target.copy_from_slice(source);
        });
    }

    /// Combine the rows of `source` inside `area` into this pixmap with `op`,
    /// which receives a row of this pixmap and the row of `source` above it.
    pub(crate) fn combine(
        &mut self,
        source: &Self,
        area: Bounds,
        mut op: impl FnMut(&mut [PremulRgba8], &[PremulRgba8]),
    ) {
        let area = area.intersect(self.bounds).intersect(source.bounds);
        let width = usize::try_from(area.width()).unwrap();
        for y in area.y0..area.y1 {
            let from = source.index(area.x0, y);
            let to = self.index(area.x0, y);
            op(
                &mut self.pixmap.data_mut()[to..to + width],
                &source.pixmap.data()[from..from + width],
            );
        }
    }

    /// Set the pixels of this pixmap, whose grid is `levels` steps of two
    /// coarser than that of `source`, that `source` covers to the average of
    /// the pixels of `source` they cover, which lines up with them.
    pub(crate) fn average(&mut self, source: &Self, levels: u32) {
        let width = usize::try_from(source.bounds.width()).unwrap();
        let coarse = source.bounds.coarsen(levels);
        let mut sums = vec![[0_u32; 4]; width >> levels];
        for (row, line) in source.pixmap.data().chunks_exact(width).enumerate() {
            for (x, pixel) in line.iter().enumerate() {
                let sum = &mut sums[x >> levels];
                for (sum, value) in sum.iter_mut().zip([pixel.r, pixel.g, pixel.b, pixel.a]) {
                    *sum += u32::from(value);
                }
            }
            if (row + 1) % (1 << levels) == 0 {
                let y = coarse.y0 + i64::try_from(row >> levels).unwrap();
                for (x, sum) in (coarse.x0..).zip(&mut sums) {
                    self.set_average(x, y, sum.map(u128::from), levels);
                    *sum = [0; 4];
                }
            }
        }
    }

    /// Add the channels of every pixel to `sums`.
    pub(crate) fn add_to(&self, sums: &mut [u128; 4]) {
        for pixel in self.pixmap.data() {
            for (sum, value) in sums.iter_mut().zip([pixel.r, pixel.g, pixel.b, pixel.a]) {
                *sum += u128::from(value);
            }
        }
    }

    /// Set the pixel at `x`, `y` to the average of the `4^levels` pixels whose
    /// channels add up to `sums`, rounded to nearest with halves up.
    pub(crate) fn set_average(&mut self, x: i64, y: i64, sums: [u128; 4], levels: u32) {
        let [r, g, b, a] =
            sums.map(|sum| ((sum.checked_shr(2 * levels - 1).unwrap_or(0) + 1) >> 1) as u8);
        let index = self.index(x, y);
        self.pixmap.data_mut()[index] = PremulRgba8 { r, g, b, a };
    }

    fn index(&self, x: i64, y: i64) -> usize {
        let bounds = self.bounds;
        usize::try_from((y - bounds.y0) * bounds.width() + x - bounds.x0).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{Bounds, Regions, Shift};
    use alloc::vec::Vec;

    /// Merge like [`Regions::insert`], comparing with every member.
    fn insert_scanning(regions: &mut Vec<Bounds>, mut bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        while let Some(index) = regions
            .iter()
            .position(|region| region.union(bounds).area() <= region.area() + bounds.area())
        {
            bounds = bounds.union(regions.swap_remove(index));
        }
        regions.push(bounds);
    }

    #[test]
    fn shifts_add_up_exactly() {
        let sum = |values: &[f32]| {
            values
                .iter()
                .fold(Shift::default(), |sum, &value| sum.add(Shift::of(value)))
                .saturated()
        };
        let big = 2.0_f32.powi(127);
        assert_eq!(sum(&[f32::MAX, 1.0, -f32::MAX]), 1);
        assert_eq!(sum(&[-f32::MAX, -3.0, f32::MAX, f32::MAX, -f32::MAX]), -3);
        assert_eq!(sum(&[big, -f32::MAX, big, -(2.0_f32.powi(104))]), 0);
        assert_eq!(sum(&[f32::MAX, f32::MAX]), i64::MAX);
        assert_eq!(sum(&[-f32::MAX, -f32::MAX, -f32::MAX]), -i64::MAX);
        assert_eq!(sum(&[-1.5e9, 2.5e9]), 1_000_000_000);
        assert_eq!(sum(&[9.0e18, 9.0e18, -9.0e18]), 9_000_000_202_358_128_640);
    }

    #[test]
    fn regions_merge_like_a_scan_of_every_member() {
        let mut state = 0x2545_f491_u64;
        let mut random = |range: i64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % range.unsigned_abs()) as i64
        };
        for round in 0..200 {
            let spread = [16, 64, 256, 1 << 40][round % 4];
            let mut regions = Regions::default();
            let mut scanned = Vec::new();
            for _ in 0..300 {
                let bounds = if round % 5 == 4 {
                    // Tiles, which merge where they share an edge.
                    let (x, y) = (4 * random(16), 4 * random(16));
                    Bounds::new(x, y, x + 4 + 4 * random(2), y + 4 + 4 * random(2))
                } else {
                    let (x, y) = (random(spread) - spread / 2, random(spread) - spread / 2);
                    let size = [1, 4, 16, 64, 1 << 20][random(5) as usize];
                    Bounds::new(x, y, x + 1 + random(size), y + 1 + random(size))
                };
                regions.insert(bounds);
                insert_scanning(&mut scanned, bounds);
            }
            assert_eq!(regions.iter().collect::<Vec<_>>(), scanned, "round {round}");
        }
    }
}
