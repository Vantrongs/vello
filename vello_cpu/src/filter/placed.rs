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

/// Return the width of the pixels of the grid `level` steps of two coarser
/// than the root target's, in root target pixels.
pub(crate) fn pixel_size(level: u32) -> f64 {
    (0..level).fold(1.0, |size, _| size * 2.0)
}

/// A whole number of pixels, `high * 2^64 + low` with `low` in `0..2^64`.
///
/// Filters move their output by up to `f32::MAX` pixels, past `2^127`, so a
/// copy that nested filters move away and back lies beyond `i128` before it
/// returns. Coordinates hold every such copy exactly, so opposite moves cancel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Coord {
    high: i128,
    low: u64,
}

impl Coord {
    pub(crate) const ZERO: Self = Self { high: 0, low: 0 };
    const ONE: Self = Self { high: 0, low: 1 };
    const MIN: Self = Self {
        high: i128::MIN,
        low: 0,
    };
    const MAX: Self = Self {
        high: i128::MAX,
        low: u64::MAX,
    };
    /// The bound of the coordinates of [`Bounds`] in each pixel grid, `2^189`.
    /// Sums of two coordinates and of a coordinate and a filter's offset or
    /// padding stay below [`Self::BEYOND`].
    const LIMIT: Self = Self {
        high: 1 << 125,
        low: 0,
    };
    /// `2^190`, where results past the domain saturate.
    const BEYOND: Self = Self {
        high: 1 << 126,
        low: 0,
    };

    const fn of_i128(value: i128) -> Self {
        Self {
            high: value >> 64,
            low: value as u64,
        }
    }

    /// Return the whole number `value`, saturated past the domain.
    pub(crate) fn of_f64(value: f64) -> Self {
        let exact = (1_u128 << 126) as f64;
        if value.abs() < exact {
            Self::of_i128(value as i128)
        } else if value.abs() < exact * (1_u128 << 64) as f64 {
            // Such a large `f64` is a multiple of `2^74`.
            Self {
                high: (value / (1_u128 << 64) as f64) as i128,
                low: 0,
            }
        } else if value > 0.0 {
            Self::BEYOND
        } else {
            -Self::BEYOND
        }
    }

    pub(crate) fn to_f64(self) -> f64 {
        match self.high.checked_mul(1 << 64) {
            Some(high) => (high | i128::from(self.low)) as f64,
            // The low part lies below the precision of the result.
            None => self.high as f64 * (1_u128 << 64) as f64,
        }
    }

    /// Return the number if it fits `i64`.
    pub(crate) fn to_i64(self) -> Option<i64> {
        let value = self.high.checked_mul(1 << 64)? | i128::from(self.low);
        i64::try_from(value).ok()
    }

    /// Return the number clamped to `±2^62`, which lies beyond any pixmap
    /// placed in the `i64` domain while sums with its lengths still fit.
    pub(crate) fn clamped_i64(self) -> i64 {
        let bound = Self::of_i128(1 << 62);
        self.clamp(-bound, bound).to_i64().unwrap()
    }

    /// Return the largest whole number at most `self / 2^levels`.
    pub(crate) fn shr(self, levels: u32) -> Self {
        match levels {
            0 => self,
            1..64 => Self {
                high: self.high >> levels,
                low: (self.low >> levels) | ((self.high as u64) << (64 - levels)),
            },
            _ => Self::of_i128(self.high >> (levels - 64).min(127)),
        }
    }

    /// Return `self * 2^levels`, saturated past the domain.
    pub(crate) fn shl(self, levels: u32) -> Self {
        let bound = Self::BEYOND.shr(levels);
        if self == Self::ZERO || levels == 0 {
            self
        } else if self >= bound {
            Self::BEYOND
        } else if self <= -bound {
            -Self::BEYOND
        } else if levels < 64 {
            Self {
                high: (self.high << levels) | i128::from(self.low >> (64 - levels)),
                low: self.low << levels,
            }
        } else {
            // The number is below `2^126`.
            Self::of_i128(((self.high << 64) | i128::from(self.low)) << (levels - 64))
        }
    }

    /// Return the number modulo `2^levels`, for `levels` up to 128.
    pub(crate) fn low_bits(self, levels: u32) -> u128 {
        let value = ((self.high as u128) << 64) | u128::from(self.low);
        value
            & u128::MAX
                .checked_shr(128_u32.saturating_sub(levels))
                .unwrap_or(0)
    }

    /// Return the number of significant bits of a number that is not negative.
    fn bits(self) -> u32 {
        if self.high > 0 {
            192 - self.high.leading_zeros()
        } else {
            u64::BITS - self.low.leading_zeros()
        }
    }

    /// Return the number of a length, saturated at `u128::MAX`.
    fn length(self) -> u128 {
        if self.high >= 1 << 64 {
            u128::MAX
        } else {
            self.low_bits(128)
        }
    }
}

impl From<i64> for Coord {
    fn from(value: i64) -> Self {
        Self::of_i128(value.into())
    }
}

impl From<i32> for Coord {
    fn from(value: i32) -> Self {
        Self::of_i128(value.into())
    }
}

impl From<u64> for Coord {
    fn from(value: u64) -> Self {
        Self::of_i128(value.into())
    }
}

impl core::ops::Add for Coord {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let low = u128::from(self.low) + u128::from(other.low);
        Self {
            high: self.high.saturating_add(other.high) + (low >> 64) as i128,
            low: low as u64,
        }
    }
}

impl core::ops::Neg for Coord {
    type Output = Self;

    fn neg(self) -> Self {
        if self.low == 0 {
            Self {
                high: -self.high,
                low: 0,
            }
        } else {
            Self {
                high: -self.high - 1,
                low: self.low.wrapping_neg(),
            }
        }
    }
}

impl core::ops::Sub for Coord {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

/// A point or offset in a pixel grid.
pub(crate) type Position = (Coord, Coord);

/// A rectangle in a pixel grid, with coordinates in `-2^189..=2^189`, see
/// [`Coord`]: whatever lies outside that domain is cut off. Empty rectangles
/// are normalized to [`Bounds::EMPTY`], so their width and height are never
/// negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bounds {
    pub(crate) x0: Coord,
    pub(crate) y0: Coord,
    pub(crate) x1: Coord,
    pub(crate) y1: Coord,
}

impl Bounds {
    pub(crate) const EMPTY: Self = Self {
        x0: Coord::ZERO,
        y0: Coord::ZERO,
        x1: Coord::ZERO,
        y1: Coord::ZERO,
    };

    pub(crate) fn new(
        x0: impl Into<Coord>,
        y0: impl Into<Coord>,
        x1: impl Into<Coord>,
        y1: impl Into<Coord>,
    ) -> Self {
        let clamp = |value: Coord| value.clamp(-Coord::LIMIT, Coord::LIMIT);
        let (x0, y0, x1, y1) = (
            clamp(x0.into()),
            clamp(y0.into()),
            clamp(x1.into()),
            clamp(y1.into()),
        );
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
            return Self::new(-Coord::LIMIT, -Coord::LIMIT, Coord::LIMIT, Coord::LIMIT);
        }
        Self::new(
            Coord::of_f64(rect.x0.floor()),
            Coord::of_f64(rect.y0.floor()),
            Coord::of_f64(rect.x1.ceil()),
            Coord::of_f64(rect.y1.ceil()),
        )
    }

    /// Return the bounds of a viewport of the given size.
    pub(crate) fn of_viewport(width: u16, height: u16) -> Self {
        Self::new(0, 0, i64::from(width), i64::from(height))
    }

    /// Return the bounds of `rect`, which is relative to `origin`.
    pub(crate) fn of_rect(rect: RectU16, origin: Position) -> Self {
        let place = |value: u16, origin: Coord| origin + i64::from(value).into();
        Self::new(
            place(rect.x0, origin.0),
            place(rect.y0, origin.1),
            place(rect.x1, origin.0),
            place(rect.y1, origin.1),
        )
    }

    pub(crate) fn is_empty(self) -> bool {
        self == Self::EMPTY
    }

    pub(crate) fn origin(self) -> Position {
        (self.x0, self.y0)
    }

    pub(crate) fn width(self) -> Coord {
        self.x1 - self.x0
    }

    pub(crate) fn height(self) -> Coord {
        self.y1 - self.y0
    }

    /// Return the width and height of bounds that fit `u32`, as a pixmap's are.
    pub(crate) fn size(self) -> (u32, u32) {
        let size = |length: Coord| {
            length
                .to_i64()
                .and_then(|length| u32::try_from(length).ok())
                .expect("filter pixmaps fit the u32 domain")
        };
        (size(self.width()), size(self.height()))
    }

    /// Return the area, saturated at `u128::MAX`.
    pub(crate) fn area(self) -> u128 {
        self.width().length().saturating_mul(self.height().length())
    }

    pub(crate) fn translate(self, (dx, dy): Position) -> Self {
        if self.is_empty() {
            return self;
        }
        Self::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }

    pub(crate) fn expand(self, padding: PaddingU64) -> Self {
        if self.is_empty() {
            return self;
        }
        Self::new(
            self.x0 - padding.left.into(),
            self.y0 - padding.top.into(),
            self.x1 + padding.right.into(),
            self.y1 + padding.bottom.into(),
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
        const {
            assert!(
                Tile::WIDTH.is_power_of_two() && Tile::HEIGHT.is_power_of_two(),
                "tiles are snapped to by shifting"
            );
        }
        let (width, height) = (Tile::WIDTH.trailing_zeros(), Tile::HEIGHT.trailing_zeros());
        let x = self.coarsen_axes(width, 0).refine_axes(width, 0);
        x.coarsen_axes(0, height).refine_axes(0, height)
    }

    /// Return the bounds of the pixels `levels` times coarser by a factor of
    /// two that cover these bounds, on a grid aligned with this one at zero.
    pub(crate) fn coarsen(self, levels: u32) -> Self {
        self.coarsen_axes(levels, levels)
    }

    fn coarsen_axes(self, x: u32, y: u32) -> Self {
        if self.is_empty() {
            return self;
        }
        let end = |value: Coord, levels: u32| -(-value).shr(levels);
        Self::new(
            self.x0.shr(x),
            self.y0.shr(y),
            end(self.x1, x),
            end(self.y1, y),
        )
    }

    /// Return the bounds of these pixels on a grid `levels` times finer by a
    /// factor of two, as far as that grid's domain reaches.
    pub(crate) fn refine(self, levels: u32) -> Self {
        self.refine_axes(levels, levels)
    }

    fn refine_axes(self, x: u32, y: u32) -> Self {
        Self::new(
            self.x0.shl(x),
            self.y0.shl(y),
            self.x1.shl(x),
            self.y1.shl(y),
        )
    }

    /// Convert bounds relative to `origin` into a `u16` rectangle, clipping the
    /// exclusive edge to the `u16` domain.
    pub(crate) fn to_rect_u16(self, origin: Position) -> RectU16 {
        if self.is_empty() {
            return RectU16::ZERO;
        }
        let local = self.translate((-origin.0, -origin.1));
        let narrow = |value: Coord| {
            u16::try_from(value.min(i64::from(u16::MAX).into()).to_i64().unwrap()).unwrap()
        };
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
    members: BTreeMap<(u32, Coord, Coord, usize), Bounds>,
    /// The classes that may hold members, as bits.
    classes: [u64; 3],
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

    fn key(id: usize, bounds: Bounds) -> (u32, Coord, Coord, usize) {
        let side = bounds.width().max(bounds.height());
        let class = (side - Coord::ONE).bits();
        (class, bounds.y0.shr(class), bounds.x0.shr(class), id)
    }

    /// Add `bounds` with `id`, unless they are empty.
    pub(crate) fn insert(&mut self, id: usize, bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        let key = Self::key(id, bounds);
        self.classes[key.0 as usize / 64] |= 1 << (key.0 % 64);
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
        let classes = (0..192)
            .filter(|&class: &u32| self.classes[class as usize / 64] & (1 << (class % 64)) != 0);
        let candidates = classes.flat_map(move |class| {
            let first = |value: Coord| (value - Coord::ONE.shl(class)).shr(class);
            let (columns, rows) = (
                first(query.x0)..=query.x1.shr(class),
                first(query.y0)..=query.y1.shr(class),
            );
            // A tall query scans the members of its rows; a short one only
            // the cells it can touch.
            let short = *rows.end() - *rows.start() < 4.into();
            let bands = if short {
                core::iter::successors(Some(*rows.start()), |&row| {
                    (row < *rows.end()).then(|| row + Coord::ONE)
                })
                .map(|row| {
                    (class, row, *columns.start(), 0)..=(class, row, *columns.end(), usize::MAX)
                })
                .collect::<Vec<_>>()
            } else {
                alloc::vec![
                    (class, *rows.start(), Coord::MIN, 0)
                        ..=(class, *rows.end(), Coord::MAX, usize::MAX)
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
            .filter(|(_, region)| {
                region.union(bounds).area() <= region.area().saturating_add(bounds.area())
            })
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
        let (width, height) = bounds.size();
        Self {
            pixmap: FilterPixmap::new(width, height),
            bounds,
        }
    }

    pub(crate) fn translate(self, offset: Position) -> Self {
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
        let (width, height) = area.size();
        let width = usize::try_from(width).unwrap();
        for row in 0..height {
            let from = source.index(area.origin(), row);
            let to = self.index(area.origin(), row);
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
        let width = usize::try_from(source.bounds.size().0).unwrap();
        let start = self.index(source.bounds.coarsen(levels).origin(), 0);
        let stride = usize::try_from(self.bounds.size().0).unwrap();
        let mut sums = vec![[0_u32; 4]; width >> levels];
        for (row, line) in source.pixmap.data().chunks_exact(width).enumerate() {
            for (x, pixel) in line.iter().enumerate() {
                let sum = &mut sums[x >> levels];
                for (sum, value) in sum.iter_mut().zip([pixel.r, pixel.g, pixel.b, pixel.a]) {
                    *sum += u32::from(value);
                }
            }
            if (row + 1) % (1 << levels) == 0 {
                let start = start + (row >> levels) * stride;
                for (index, sum) in (start..).zip(&mut sums) {
                    self.set_average(index, sum.map(u128::from), levels);
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

    /// Set the pixel at `index` to the average of the `4^levels` pixels whose
    /// channels add up to `sums`, rounded to nearest with halves up.
    pub(crate) fn set_average(&mut self, index: usize, sums: [u128; 4], levels: u32) {
        let [r, g, b, a] =
            sums.map(|sum| ((sum.checked_shr(2 * levels - 1).unwrap_or(0) + 1) >> 1) as u8);
        self.pixmap.data_mut()[index] = PremulRgba8 { r, g, b, a };
    }

    /// Return the index of the pixel `row` rows below `position`.
    fn index(&self, (x, y): Position, row: u32) -> usize {
        let offset = |value: Coord, start: Coord| {
            usize::try_from((value - start).to_i64().unwrap()).unwrap()
        };
        let width = usize::try_from(self.bounds.size().0).unwrap();
        (offset(y, self.bounds.y0) + usize::try_from(row).unwrap()) * width
            + offset(x, self.bounds.x0)
    }
}

#[cfg(test)]
mod tests {
    use super::{Bounds, Coord, Regions};
    use alloc::vec::Vec;

    /// Merge like [`Regions::insert`], comparing with every member.
    fn insert_scanning(regions: &mut Vec<Bounds>, mut bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        while let Some(index) = regions.iter().position(|region| {
            region.union(bounds).area() <= region.area().saturating_add(bounds.area())
        }) {
            bounds = bounds.union(regions.swap_remove(index));
        }
        regions.push(bounds);
    }

    #[test]
    fn offsets_add_up_exactly() {
        let sum = |values: &[f32]| {
            values.iter().fold(Coord::ZERO, |sum, &value| {
                sum + Coord::of_f64(f64::from(value))
            })
        };
        let big = 2.0_f32.powi(127);
        assert_eq!(sum(&[f32::MAX, 1.0, -f32::MAX]), Coord::from(1));
        assert_eq!(
            sum(&[-f32::MAX, -3.0, f32::MAX, f32::MAX, -f32::MAX]),
            Coord::from(-3)
        );
        assert_eq!(
            sum(&[big, -f32::MAX, big, -(2.0_f32.powi(104))]),
            Coord::ZERO
        );
        assert_eq!(
            sum(&[f32::MAX, f32::MAX]) - sum(&[f32::MAX]),
            sum(&[f32::MAX])
        );
        assert!(sum(&[f32::MAX, f32::MAX, f32::MAX]) > sum(&[f32::MAX, f32::MAX]));
        assert_eq!(sum(&[-1.5e9, 2.5e9]), Coord::from(1_000_000_000));
        assert_eq!(
            sum(&[9.0e18, 9.0e18, -9.0e18]).to_f64(),
            9_000_000_202_358_128_640.0
        );
    }

    #[test]
    fn coordinates_compute_like_wide_integers() {
        let mut state = 0x2545_f491_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..10_000 {
            // Values of every magnitude below `2^100`, which `i128` holds
            // with their products by `2^20` and sums.
            let bits = random() % 100;
            let value = (i128::from(random()) << 64 | i128::from(random())) >> (127 - bits);
            let other = (i128::from(random()) << 64 | i128::from(random())) >> (127 - bits);
            let coord = Coord::of_i128;
            let levels = u32::try_from(random() % 140).unwrap();
            assert_eq!(coord(value) + coord(other), coord(value + other));
            assert_eq!(coord(value) - coord(other), coord(value - other));
            assert_eq!(coord(value) < coord(other), value < other);
            assert_eq!(coord(value).shr(levels), coord(value >> levels.min(127)));
            if levels <= 20 {
                assert_eq!(coord(value).shl(levels), coord(value << levels));
            }
            if levels <= 126 {
                let modulo = value.rem_euclid(1 << levels) as u128;
                assert_eq!(coord(value).low_bits(levels), modulo);
            }
            assert_eq!(coord(value).to_i64(), i64::try_from(value).ok());
            assert_eq!(coord(value).to_f64(), value as f64);
        }
        let far = Coord::of_f64(f64::from(f32::MAX));
        assert_eq!(far.shl(62).shr(62), far);
        assert_eq!(far.shl(63), Coord::BEYOND);
        assert_eq!((-far).shl(63), -Coord::BEYOND);
        assert_eq!(far.shr(200), Coord::ZERO);
        assert_eq!((-far).shr(200), Coord::from(-1));
        assert_eq!(Coord::of_f64(f64::MAX), Coord::BEYOND);
        assert_eq!(Coord::of_f64(-1.0e60), -Coord::BEYOND);
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
