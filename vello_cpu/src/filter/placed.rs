// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter pixmaps placed in a source space that can exceed the `u16` domain.

use crate::filter::pixmap::FilterPixmap;
use crate::kurbo::Rect;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;
use vello_common::geometry::RectU16;
use vello_common::peniko::color::PremulRgba8;
use vello_common::tile::Tile;

/// A whole number of pixels.
///
/// Filters move their output by up to `f32::MAX` pixels times the transform,
/// and nested filters move copies away and back, so coordinates are exact
/// integers of any size: opposite moves always cancel and nothing is cut off.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Coord(BigInt);

impl Coord {
    pub(crate) const ZERO: Self = Self(BigInt::ZERO);

    pub(crate) fn new(value: BigInt) -> Self {
        Self(value)
    }

    /// Return the largest whole number at most `value`, which is finite.
    pub(crate) fn floor_of(value: f64) -> Self {
        debug_assert!(value.is_finite(), "only finite values have a floor");
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1 << 52) - 1);
        let (mantissa, exponent) = if exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), exponent - 1075)
        };
        let mut mantissa = BigInt::from(mantissa);
        if value.is_sign_negative() {
            mantissa = -mantissa;
        }
        Self(if exponent >= 0 {
            mantissa << exponent as u64
        } else {
            // Shifting right rounds towards negative infinity.
            mantissa >> exponent.unsigned_abs()
        })
    }

    /// Return the smallest whole number at least `value`, which is finite.
    pub(crate) fn ceil_of(value: f64) -> Self {
        -Self::floor_of(-value)
    }

    /// Return the nearest `f64`.
    pub(crate) fn to_f64(&self) -> f64 {
        self.0.to_f64().unwrap_or(f64::NAN)
    }

    /// Return the number if it fits `i64`.
    pub(crate) fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    /// Return the number clamped to `±2^62`, which lies beyond any pixmap
    /// placed in the `i64` domain while sums with its lengths still fit.
    pub(crate) fn clamped_i64(&self) -> i64 {
        const BOUND: i64 = 1 << 62;
        self.0
            .to_i64()
            .unwrap_or(if self.0.is_negative() { -BOUND } else { BOUND })
            .clamp(-BOUND, BOUND)
    }

    /// Return the number as a length or offset within a pixmap, which fits
    /// `usize` wherever pixmaps are addressed.
    pub(crate) fn to_usize(&self) -> usize {
        self.0
            .to_usize()
            .expect("offsets within a pixmap fit usize")
    }

    /// Return the largest whole number at most `self / 2^levels`.
    pub(crate) fn shr(&self, levels: u32) -> Self {
        Self(&self.0 >> levels)
    }

    /// Return `self * 2^levels`.
    pub(crate) fn shl(&self, levels: u32) -> Self {
        Self(&self.0 << levels)
    }

    pub(crate) fn is_odd(&self) -> bool {
        self.0.bit(0)
    }

    pub(crate) fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    /// Return the number of significant bits of a number that is not negative.
    fn bits(&self) -> u64 {
        self.0.bits()
    }

    /// Return the product of two numbers.
    pub(crate) fn mul(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    /// Return the number modulo `modulus`, which is positive, in `0..modulus`.
    pub(crate) fn rem_euclid(&self, modulus: &Self) -> Self {
        Self(self.0.mod_floor(&modulus.0))
    }

    /// Return the largest whole number at most `self / divisor`, which is positive.
    pub(crate) fn div_floor(&self, divisor: &Self) -> Self {
        Self(self.0.div_floor(&divisor.0))
    }
}

trait IsNegative {
    fn is_negative(&self) -> bool;
}

impl IsNegative for BigInt {
    fn is_negative(&self) -> bool {
        self.sign() == num_bigint::Sign::Minus
    }
}

impl From<i64> for Coord {
    fn from(value: i64) -> Self {
        Self(value.into())
    }
}

impl From<i32> for Coord {
    fn from(value: i32) -> Self {
        Self(value.into())
    }
}

impl From<u32> for Coord {
    fn from(value: u32) -> Self {
        Self(value.into())
    }
}

impl From<u16> for Coord {
    fn from(value: u16) -> Self {
        Self(value.into())
    }
}

impl From<u64> for Coord {
    fn from(value: u64) -> Self {
        Self(value.into())
    }
}

impl From<usize> for Coord {
    fn from(value: usize) -> Self {
        Self(value.into())
    }
}

macro_rules! arithmetic {
    ($trait:ident, $method:ident) => {
        impl core::ops::$trait<Coord> for Coord {
            type Output = Coord;

            fn $method(self, other: Coord) -> Coord {
                Coord(core::ops::$trait::$method(self.0, other.0))
            }
        }

        impl core::ops::$trait<&Coord> for Coord {
            type Output = Coord;

            fn $method(self, other: &Coord) -> Coord {
                Coord(core::ops::$trait::$method(self.0, &other.0))
            }
        }

        impl core::ops::$trait<Coord> for &Coord {
            type Output = Coord;

            fn $method(self, other: Coord) -> Coord {
                Coord(core::ops::$trait::$method(&self.0, other.0))
            }
        }

        impl core::ops::$trait<&Coord> for &Coord {
            type Output = Coord;

            fn $method(self, other: &Coord) -> Coord {
                Coord(core::ops::$trait::$method(&self.0, &other.0))
            }
        }

        impl core::ops::$trait<i64> for &Coord {
            type Output = Coord;

            fn $method(self, other: i64) -> Coord {
                Coord(core::ops::$trait::$method(&self.0, BigInt::from(other)))
            }
        }

        impl core::ops::$trait<i64> for Coord {
            type Output = Coord;

            fn $method(self, other: i64) -> Coord {
                Coord(core::ops::$trait::$method(self.0, BigInt::from(other)))
            }
        }
    };
}

arithmetic!(Add, add);
arithmetic!(Sub, sub);

impl core::ops::Neg for Coord {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl core::ops::Neg for &Coord {
    type Output = Coord;

    fn neg(self) -> Coord {
        Coord(-&self.0)
    }
}

/// A point or offset in a pixel grid.
pub(crate) type Position = (Coord, Coord);

/// How far a rectangle reaches past another on each side, in whole pixels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Padding {
    pub(crate) left: Coord,
    pub(crate) top: Coord,
    pub(crate) right: Coord,
    pub(crate) bottom: Coord,
}

impl Padding {
    /// Return the padding of `amount` on every side.
    pub(crate) fn uniform(amount: Coord) -> Self {
        Self {
            left: amount.clone(),
            top: amount.clone(),
            right: amount.clone(),
            bottom: amount,
        }
    }
}

/// A rectangle in a pixel grid. Empty rectangles are normalized to
/// [`Bounds::EMPTY`], so their width and height are never negative.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        let (x0, y0, x1, y1) = (x0.into(), y0.into(), x1.into(), y1.into());
        if x0 >= x1 || y0 >= y1 {
            return Self::EMPTY;
        }
        Self { x0, y0, x1, y1 }
    }

    /// Return the part of these bounds in the pixels that `rect` touches; a
    /// side of `rect` that is not finite does not limit them.
    pub(crate) fn clip(&self, rect: Rect) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let bound = |value: f64, round: fn(f64) -> Coord| value.is_finite().then(|| round(value));
        let x0 =
            bound(rect.x0, Coord::floor_of).map_or(self.x0.clone(), |x| x.max(self.x0.clone()));
        let y0 =
            bound(rect.y0, Coord::floor_of).map_or(self.y0.clone(), |y| y.max(self.y0.clone()));
        let x1 = bound(rect.x1, Coord::ceil_of).map_or(self.x1.clone(), |x| x.min(self.x1.clone()));
        let y1 = bound(rect.y1, Coord::ceil_of).map_or(self.y1.clone(), |y| y.min(self.y1.clone()));
        Self::new(x0, y0, x1, y1)
    }

    /// Return the bounds of the pixels that `rect` touches, or `None` if it
    /// is not finite.
    pub(crate) fn covering(rect: Rect) -> Option<Self> {
        [rect.x0, rect.y0, rect.x1, rect.y1]
            .iter()
            .all(|value| value.is_finite())
            .then(|| {
                Self::new(
                    Coord::floor_of(rect.x0),
                    Coord::floor_of(rect.y0),
                    Coord::ceil_of(rect.x1),
                    Coord::ceil_of(rect.y1),
                )
            })
    }

    /// Return the bounds of a viewport of the given size.
    pub(crate) fn of_viewport(width: u16, height: u16) -> Self {
        Self::new(0, 0, width, height)
    }

    /// Return the bounds of `rect`, which is relative to `origin`.
    pub(crate) fn of_rect(rect: RectU16, origin: &Position) -> Self {
        Self::new(
            &origin.0 + i64::from(rect.x0),
            &origin.1 + i64::from(rect.y0),
            &origin.0 + i64::from(rect.x1),
            &origin.1 + i64::from(rect.y1),
        )
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.x0 >= self.x1
    }

    pub(crate) fn origin(&self) -> Position {
        (self.x0.clone(), self.y0.clone())
    }

    pub(crate) fn width(&self) -> Coord {
        &self.x1 - &self.x0
    }

    pub(crate) fn height(&self) -> Coord {
        &self.y1 - &self.y0
    }

    /// Return the width and height of bounds that fit `u32`, as those of the
    /// pixmaps they place do by construction.
    pub(crate) fn size(&self) -> (u32, u32) {
        let size = |length: Coord| {
            length
                .to_i64()
                .and_then(|length| u32::try_from(length).ok())
                .expect("filter pixmaps fit the u32 domain")
        };
        (size(self.width()), size(self.height()))
    }

    pub(crate) fn area(&self) -> Coord {
        self.width().mul(&self.height())
    }

    pub(crate) fn translate(&self, (dx, dy): &Position) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self::new(&self.x0 + dx, &self.y0 + dy, &self.x1 + dx, &self.y1 + dy)
    }

    pub(crate) fn expand(&self, padding: &Padding) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self::new(
            &self.x0 - &padding.left,
            &self.y0 - &padding.top,
            &self.x1 + &padding.right,
            &self.y1 + &padding.bottom,
        )
    }

    pub(crate) fn intersect(&self, other: &Self) -> Self {
        Self::new(
            (&self.x0).max(&other.x0).clone(),
            (&self.y0).max(&other.y0).clone(),
            (&self.x1).min(&other.x1).clone(),
            (&self.y1).min(&other.y1).clone(),
        )
    }

    /// Return the smallest bounds containing both.
    pub(crate) fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        Self::new(
            (&self.x0).min(&other.x0).clone(),
            (&self.y0).min(&other.y0).clone(),
            (&self.x1).max(&other.x1).clone(),
            (&self.y1).max(&other.y1).clone(),
        )
    }

    /// Return whether the bounds overlap `other` or share an edge with it.
    pub(crate) fn touches(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.x0 <= other.x1
            && other.x0 <= self.x1
            && self.y0 <= other.y1
            && other.y0 <= self.y1
    }

    /// Return whether the bounds hold `other`.
    pub(crate) fn contains(&self, other: &Self) -> bool {
        other.is_empty()
            || (self.x0 <= other.x0
                && self.y0 <= other.y0
                && other.x1 <= self.x1
                && other.y1 <= self.y1)
    }

    /// Return the bounds of the tiles they touch.
    pub(crate) fn snap_to_tiles(&self) -> Self {
        const {
            assert!(
                Tile::WIDTH.is_power_of_two() && Tile::HEIGHT.is_power_of_two(),
                "tiles are snapped to by shifting"
            );
        }
        if self.is_empty() {
            return Self::EMPTY;
        }
        let (width, height) = (Tile::WIDTH.trailing_zeros(), Tile::HEIGHT.trailing_zeros());
        let down = |value: &Coord, levels: u32| value.shr(levels).shl(levels);
        let up = |value: &Coord, levels: u32| -down(&-value, levels);
        Self::new(
            down(&self.x0, width),
            down(&self.y0, height),
            up(&self.x1, width),
            up(&self.y1, height),
        )
    }

    /// Convert bounds relative to `origin`, which they lie at or below and
    /// right of, into a `u16` rectangle, clipping the exclusive edge to the
    /// `u16` domain.
    pub(crate) fn to_rect_u16(&self, origin: &Position) -> RectU16 {
        if self.is_empty() {
            return RectU16::ZERO;
        }
        let narrow = |value: &Coord, origin: &Coord| {
            let local = (value - origin).to_i64().unwrap_or(i64::MAX);
            u16::try_from(local.clamp(0, i64::from(u16::MAX))).unwrap()
        };
        RectU16::new(
            narrow(&self.x0, &origin.0),
            narrow(&self.y0, &origin.1),
            narrow(&self.x1, &origin.0),
            narrow(&self.y1, &origin.1),
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
    /// Each member by class and cell row, then by cell column and id.
    members: BTreeMap<(u64, Coord), BTreeMap<(Coord, usize), Bounds>>,
    /// The number of members of each class that has any.
    classes: BTreeMap<u64, usize>,
}

impl Index {
    fn key(id: usize, bounds: &Bounds) -> ((u64, Coord), (Coord, usize)) {
        let side = bounds.width().max(bounds.height());
        let class = (side - 1).bits();
        let levels = u32::try_from(class).expect("sides have fewer than 2^32 bits");
        ((class, bounds.y0.shr(levels)), (bounds.x0.shr(levels), id))
    }

    /// Add `bounds` with `id`, unless they are empty.
    pub(crate) fn insert(&mut self, id: usize, bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        let (row, column) = Self::key(id, &bounds);
        *self.classes.entry(row.0).or_default() += 1;
        self.members.entry(row).or_default().insert(column, bounds);
    }

    /// Remove the member `bounds` with `id`.
    pub(crate) fn remove(&mut self, id: usize, bounds: &Bounds) {
        let (row, column) = Self::key(id, bounds);
        let Some(columns) = self.members.get_mut(&row) else {
            return;
        };
        if columns.remove(&column).is_none() {
            return;
        }
        if columns.is_empty() {
            self.members.remove(&row);
        }
        let count = self.classes.get_mut(&row.0).unwrap();
        *count -= 1;
        if *count == 0 {
            self.classes.remove(&row.0);
        }
    }

    /// Return the ids and bounds of the members that overlap `query` or share
    /// an edge with it, in no particular order.
    pub(crate) fn touching<'a>(
        &'a self,
        query: &'a Bounds,
    ) -> impl Iterator<Item = (usize, &'a Bounds)> + 'a {
        self.classes
            .keys()
            .flat_map(move |&class| {
                let levels = u32::try_from(class).unwrap();
                let first = |value: &Coord| (value - Coord::from(1_i64).shl(levels)).shr(levels);
                let rows = (class, first(&query.y0))..=(class, query.y1.shr(levels));
                let columns = (first(&query.x0), 0)..=(query.x1.shr(levels), usize::MAX);
                self.members
                    .range(rows)
                    .flat_map(move |(_, members)| members.range(columns.clone()))
            })
            .filter(move |(_, bounds)| bounds.touches(query))
            .map(|(&(_, id), bounds)| (id, bounds))
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
            .touching(&bounds)
            .filter(|(_, region)| region.union(&bounds).area() <= region.area() + bounds.area())
            .map(|(index, _)| index)
            .min()
        {
            bounds = bounds.union(&self.swap_remove(index));
        }
        self.index.insert(self.regions.len(), bounds.clone());
        self.regions.push(bounds);
    }

    fn swap_remove(&mut self, index: usize) -> Bounds {
        let last = self.regions.len() - 1;
        self.index.remove(index, &self.regions[index]);
        if index != last {
            self.index.remove(last, &self.regions[last]);
            self.index.insert(index, self.regions[last].clone());
        }
        self.regions.swap_remove(index)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Bounds> + '_ {
        self.regions.iter()
    }

    /// Return the regions that overlap `bounds` or share an edge with it, in
    /// the order of [`Self::iter`].
    pub(crate) fn touching(&self, bounds: &Bounds) -> impl Iterator<Item = &Bounds> + '_ {
        let mut indices: Vec<usize> = self
            .index
            .touching(bounds)
            .map(|(index, _)| index)
            .collect();
        indices.sort_unstable();
        indices.into_iter().map(|index| &self.regions[index])
    }

    /// Return whether a region overlaps `bounds`.
    pub(crate) fn overlaps(&self, bounds: &Bounds) -> bool {
        self.index
            .touching(bounds)
            .any(|(_, region)| !region.intersect(bounds).is_empty())
    }

    /// Return the bounds of the parts of the regions inside `bounds`.
    pub(crate) fn hull_within(&self, bounds: &Bounds) -> Bounds {
        self.index
            .touching(bounds)
            .fold(Bounds::EMPTY, |hull, (_, region)| {
                hull.union(&region.intersect(bounds))
            })
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

    pub(crate) fn translate(self, offset: &Position) -> Self {
        Self {
            bounds: self.bounds.translate(offset),
            ..self
        }
    }

    /// Copy the pixels of `source` inside `area`.
    pub(crate) fn copy_from(&mut self, source: &Self, area: &Bounds) {
        self.combine(source, area, |target, source| {
            target.copy_from_slice(source);
        });
    }

    /// Combine the rows of `source` inside `area` into this pixmap with `op`,
    /// which receives a row of this pixmap and the row of `source` above it.
    pub(crate) fn combine(
        &mut self,
        source: &Self,
        area: &Bounds,
        mut op: impl FnMut(&mut [PremulRgba8], &[PremulRgba8]),
    ) {
        let area = area.intersect(&self.bounds).intersect(&source.bounds);
        if area.is_empty() {
            return;
        }
        let (width, height) = area.size();
        let width = usize::try_from(width).unwrap();
        let from = source.index_of(&area.origin());
        let to = self.index_of(&area.origin());
        let (from_stride, to_stride) = (source.stride(), self.stride());
        for row in 0..usize::try_from(height).unwrap() {
            let (from, to) = (from + row * from_stride, to + row * to_stride);
            op(
                &mut self.pixmap.data_mut()[to..to + width],
                &source.pixmap.data()[from..from + width],
            );
        }
    }

    fn stride(&self) -> usize {
        usize::try_from(self.pixmap.width()).unwrap()
    }

    /// Return the index of the pixel at `position`, which lies in the pixmap.
    fn index_of(&self, (x, y): &Position) -> usize {
        (y - &self.bounds.y0).to_usize() * self.stride() + (x - &self.bounds.x0).to_usize()
    }
}

#[cfg(test)]
mod tests {
    use super::{Bounds, Coord, Regions};
    use alloc::vec::Vec;
    use num_bigint::BigInt;

    /// Merge like [`Regions::insert`], comparing with every member.
    fn insert_scanning(regions: &mut Vec<Bounds>, mut bounds: Bounds) {
        if bounds.is_empty() {
            return;
        }
        while let Some(index) = regions
            .iter()
            .position(|region| region.union(&bounds).area() <= region.area() + bounds.area())
        {
            bounds = bounds.union(&regions.swap_remove(index));
        }
        regions.push(bounds);
    }

    #[test]
    fn whole_numbers_of_floats_are_exact() {
        let big = |value: f64| Coord::floor_of(value);
        assert_eq!(big(-1.5), Coord::from(-2_i64));
        assert_eq!(Coord::ceil_of(-1.5), Coord::from(-1_i64));
        assert_eq!(Coord::ceil_of(2.0), Coord::from(2_i64));
        assert_eq!(big(f64::MIN_POSITIVE / 2.0), Coord::ZERO);
        assert_eq!(big(-f64::MIN_POSITIVE / 2.0), Coord::from(-1_i64));
        assert_eq!(
            big(f64::from(f32::MAX)),
            Coord::new(BigInt::from(0xff_ffff_u64) << 104)
        );
        let max = big(f64::MAX);
        assert_eq!(max.to_f64(), f64::MAX);
        assert_eq!((&max + &max - &max - max.clone()), Coord::ZERO);
        assert_eq!(Coord::from(-5_i64).shr(1), Coord::from(-3_i64));
        assert_eq!(Coord::from(-5_i64).shr(200), Coord::from(-1_i64));
        assert_eq!(Coord::from(5_i64).shr(200), Coord::ZERO);
        assert_eq!(
            Coord::from(1_i64).shl(64),
            Coord::new(BigInt::from(1_u128 << 64))
        );
        assert_eq!(
            Coord::from(-7_i64).rem_euclid(&Coord::from(3_i64)),
            Coord::from(2_i64)
        );
    }

    #[test]
    fn shifts_compute_like_integers_at_every_width() {
        for bits in [63_u32, 64, 65, 94, 122, 127, 128, 129, 300] {
            let value = Coord::from(1_i64).shl(bits) - 3;
            for levels in [0_u32, 1, 2, 63, 64, 65, 70, 200] {
                let expected = Coord::new(BigInt::from(2).pow(bits) - 3);
                assert_eq!(value, expected);
                let shifted = value.shl(levels);
                assert_eq!(
                    shifted,
                    Coord::new(&expected.0 * BigInt::from(2).pow(levels))
                );
                assert_eq!(shifted.shr(levels), value, "{bits} {levels}");
                assert_eq!((-&value).shl(levels).shr(levels), -&value);
            }
        }
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
                regions.insert(bounds.clone());
                insert_scanning(&mut scanned, bounds);
            }
            assert_eq!(
                regions.iter().cloned().collect::<Vec<_>>(),
                scanned,
                "round {round}"
            );
        }
    }
}
