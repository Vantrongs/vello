// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometry utilities.

use crate::kurbo::Rect;
use crate::tile::Tile;
use bytemuck::{Pod, Zeroable};
use core::ops::Add;

/// A size represented by two 16-bit unsigned integers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct SizeU16(pub [u16; 2]);

impl SizeU16 {
    /// A zero size.
    pub const ZERO: Self = Self::new(0);

    /// Create a new square size.
    pub const fn new(size: u16) -> Self {
        Self([size; 2])
    }

    /// Create a new size from its width and height.
    pub const fn from_wh(width: u16, height: u16) -> Self {
        Self([width, height])
    }

    /// The width of this size.
    pub const fn width(self) -> u16 {
        self.0[0]
    }

    /// The height of this size.
    pub const fn height(self) -> u16 {
        self.0[1]
    }

    /// Return the maximum of the two sizes.
    pub fn max(self, other: Self) -> Self {
        Self::from_wh(
            self.width().max(other.width()),
            self.height().max(other.height()),
        )
    }

    /// Return the minimum of the two sizes.
    pub fn min(self, other: Self) -> Self {
        Self::from_wh(
            self.width().min(other.width()),
            self.height().min(other.height()),
        )
    }

    /// Clamp both dimensions to the given range.
    pub fn clamp(self, min: u16, max: u16) -> Self {
        Self::from_wh(self.width().clamp(min, max), self.height().clamp(min, max))
    }

    /// Add the same value to both dimensions, returning `None` on overflow.
    pub fn checked_add(self, value: u16) -> Option<Self> {
        Some(Self::from_wh(
            self.width().checked_add(value)?,
            self.height().checked_add(value)?,
        ))
    }
}

impl From<[u16; 2]> for SizeU16 {
    fn from(value: [u16; 2]) -> Self {
        Self(value)
    }
}

impl From<(u16, u16)> for SizeU16 {
    fn from((width, height): (u16, u16)) -> Self {
        Self::from_wh(width, height)
    }
}

impl From<SizeU16> for (u16, u16) {
    fn from(size: SizeU16) -> Self {
        (size.width(), size.height())
    }
}

impl Add for SizeU16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Shouldn't overflow for our use cases.
        Self::from_wh(
            self.width().checked_add(rhs.width()).unwrap(),
            self.height().checked_add(rhs.height()).unwrap(),
        )
    }
}

impl Add<u16> for SizeU16 {
    type Output = Self;

    fn add(self, rhs: u16) -> Self::Output {
        self + Self::new(rhs)
    }
}

/// Padding for the four sides of a region.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PaddingU16 {
    /// The left padding.
    pub left: u16,
    /// The top padding.
    pub top: u16,
    /// The right padding.
    pub right: u16,
    /// The bottom padding.
    pub bottom: u16,
}

impl PaddingU16 {
    /// Padding with all sides set to zero.
    pub const ZERO: Self = Self::new(0, 0, 0, 0);

    /// Create padding from its left, top, right, and bottom amounts.
    pub const fn new(left: u16, top: u16, right: u16, bottom: u16) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}

/// An axis-aligned rectangle with `u16` coordinates, stored as two corners `(x0, y0)` and
/// `(x1, y1)`.
///
/// `(x0, y0)` is the top-left (minimum) corner and `(x1, y1)` is the bottom-right (maximum) corner.
/// The rectangle is considered to be empty when `x0 >= x1` or `y0 >= y1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RectU16 {
    /// The minimum x coordinate (left edge).
    pub x0: u16,
    /// The minimum y coordinate (top edge).
    pub y0: u16,
    /// The maximum x coordinate (right edge, exclusive).
    pub x1: u16,
    /// The maximum y coordinate (bottom edge, exclusive).
    pub y1: u16,
}

impl RectU16 {
    /// Return the outward-rounded bounds in tile units.
    ///
    /// The exclusive tile endpoint can be 16384 even though its pixel endpoint
    /// (65536) is outside the physical `u16` coordinate domain.
    #[inline(always)]
    pub const fn to_tile_bounds(self) -> Self {
        let x0 = self.x0 / Tile::WIDTH;
        let y0 = self.y0 / Tile::HEIGHT;
        if self.is_empty() {
            return Self::new(x0, y0, x0, y0);
        }
        Self::new(
            x0,
            y0,
            self.x1.div_ceil(Tile::WIDTH),
            self.y1.div_ceil(Tile::HEIGHT),
        )
    }

    /// Convert tile bounds to their intersection with the physical `u16`
    /// coordinate domain. The last physical tile may be partial: no `u16`-sized
    /// target contains the pixel at coordinate 65535.
    #[inline(always)]
    pub const fn from_tile_bounds(tiles: Self) -> Self {
        Self::new(
            tiles.x0.saturating_mul(Tile::WIDTH),
            tiles.y0.saturating_mul(Tile::HEIGHT),
            tiles.x1.saturating_mul(Tile::WIDTH),
            tiles.y1.saturating_mul(Tile::HEIGHT),
        )
    }

    /// A rectangle with all coordinates set to zero.
    pub const ZERO: Self = Self {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    /// An empty, maximally inverted rectangle, useful as a starting value for incremental union
    /// operations.
    ///
    /// Has `(x0, y0) = (u16::MAX, u16::MAX)` and `(x1, y1) = (0, 0)`.
    pub const INVERTED: Self = Self {
        x0: u16::MAX,
        y0: u16::MAX,
        x1: 0,
        y1: 0,
    };

    /// Create a new rectangle from its corner coordinates.
    #[inline(always)]
    pub const fn new(x0: u16, y0: u16, x1: u16, y1: u16) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The width of the rectangle (`x1 - x0`), saturating at zero.
    #[inline(always)]
    pub const fn width(self) -> u16 {
        self.x1.saturating_sub(self.x0)
    }

    /// The height of the rectangle (`y1 - y0`), saturating at zero.
    #[inline(always)]
    pub const fn height(self) -> u16 {
        self.y1.saturating_sub(self.y0)
    }

    /// Returns `true` if the rectangle has zero area (`x0 >= x1` or `y0 >= y1`).
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    /// Check if a point `(x, y)` is contained within this rectangle.
    ///
    /// Returns `true` if `x0 <= x < x1` and `y0 <= y < y1`.
    #[inline(always)]
    pub const fn contains(self, x: u16, y: u16) -> bool {
        (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)
    }

    /// Compute the intersection of two rectangles.
    ///
    /// The result may have zero area if the rectangles do not overlap, but is never inverted.
    #[inline(always)]
    pub const fn intersect(self, other: Self) -> Self {
        let x0 = const_max(self.x0, other.x0);
        let y0 = const_max(self.y0, other.y0);
        let x1 = const_min(self.x1, other.x1);
        let y1 = const_min(self.y1, other.y1);

        Self::new(x0, y0, const_max(x1, x0), const_max(y1, y0))
    }

    /// Expand this rectangle by the given left, top, right, and bottom padding.
    #[inline(always)]
    pub const fn expand(self, padding: PaddingU16) -> Self {
        Self {
            x0: self.x0.saturating_sub(padding.left),
            y0: self.y0.saturating_sub(padding.top),
            x1: self.x1.saturating_add(padding.right),
            y1: self.y1.saturating_add(padding.bottom),
        }
    }

    /// Return this rectangle relative to `origin`, clamping negative coordinates to zero.
    #[inline(always)]
    pub fn relative_to_origin(self, origin: (u16, u16)) -> Self {
        self.shift((-(origin.0 as i32), -(origin.1 as i32)))
    }

    /// Return a shifted version of the rectangle, clamping negative coordinates to zero.
    #[inline]
    pub fn shift(self, shift: (i32, i32)) -> Self {
        Self {
            x0: (self.x0 as i32)
                .saturating_add(shift.0)
                .clamp(0, u16::MAX as i32) as u16,
            y0: (self.y0 as i32)
                .saturating_add(shift.1)
                .clamp(0, u16::MAX as i32) as u16,
            x1: (self.x1 as i32)
                .saturating_add(shift.0)
                .clamp(0, u16::MAX as i32) as u16,
            y1: (self.y1 as i32)
                .saturating_add(shift.1)
                .clamp(0, u16::MAX as i32) as u16,
        }
    }

    /// Expand this rectangle to also cover `other` (union in place).
    ///
    /// The union of `self` with a [`Self::INVERTED`] returns `self`.
    #[inline(always)]
    pub const fn union(&mut self, other: Self) {
        self.x0 = const_min(self.x0, other.x0);
        self.y0 = const_min(self.y0, other.y0);
        self.x1 = const_max(self.x1, other.x1);
        self.y1 = const_max(self.y1, other.y1);
    }

    /// Return the rect as a [`Rect`].
    pub fn as_rect(self) -> Rect {
        Rect::new(
            self.x0 as f64,
            self.y0 as f64,
            self.x1 as f64,
            self.y1 as f64,
        )
    }
}

impl From<RectU16> for SizeU16 {
    fn from(rect: RectU16) -> Self {
        Self::from_wh(rect.width(), rect.height())
    }
}

#[inline(always)]
const fn const_max(a: u16, b: u16) -> u16 {
    if a > b { a } else { b }
}

#[inline(always)]
const fn const_min(a: u16, b: u16) -> u16 {
    if a < b { a } else { b }
}

/// A size represented by two 32-bit unsigned integers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct SizeU32(pub [u32; 2]);

impl SizeU32 {
    /// A zero size.
    pub const ZERO: Self = Self::new(0);

    /// Create a new square size.
    pub const fn new(size: u32) -> Self {
        Self([size; 2])
    }

    /// Create a new size from its width and height.
    pub const fn from_wh(width: u32, height: u32) -> Self {
        Self([width, height])
    }

    /// The width of this size.
    pub const fn width(self) -> u32 {
        self.0[0]
    }

    /// The height of this size.
    pub const fn height(self) -> u32 {
        self.0[1]
    }

    /// Return the maximum of the two sizes.
    pub fn max(self, other: Self) -> Self {
        Self::from_wh(
            self.width().max(other.width()),
            self.height().max(other.height()),
        )
    }

    /// Return the minimum of the two sizes.
    pub fn min(self, other: Self) -> Self {
        Self::from_wh(
            self.width().min(other.width()),
            self.height().min(other.height()),
        )
    }

    /// Clamp both dimensions to the given range.
    pub fn clamp(self, min: u32, max: u32) -> Self {
        Self::from_wh(self.width().clamp(min, max), self.height().clamp(min, max))
    }

    /// Add the same value to both dimensions, returning `None` on overflow.
    pub fn checked_add(self, value: u32) -> Option<Self> {
        Some(Self::from_wh(
            self.width().checked_add(value)?,
            self.height().checked_add(value)?,
        ))
    }
}

impl From<[u32; 2]> for SizeU32 {
    fn from(value: [u32; 2]) -> Self {
        Self(value)
    }
}

impl From<(u32, u32)> for SizeU32 {
    fn from((width, height): (u32, u32)) -> Self {
        Self::from_wh(width, height)
    }
}

impl From<SizeU32> for (u32, u32) {
    fn from(size: SizeU32) -> Self {
        (size.width(), size.height())
    }
}

impl Add for SizeU32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Shouldn't overflow for our use cases.
        Self::from_wh(
            self.width().checked_add(rhs.width()).unwrap(),
            self.height().checked_add(rhs.height()).unwrap(),
        )
    }
}

impl Add<u32> for SizeU32 {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        self + Self::new(rhs)
    }
}

/// Padding for the four sides of a region.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PaddingU32 {
    /// The left padding.
    pub left: u32,
    /// The top padding.
    pub top: u32,
    /// The right padding.
    pub right: u32,
    /// The bottom padding.
    pub bottom: u32,
}

impl PaddingU32 {
    /// Padding with all sides set to zero.
    pub const ZERO: Self = Self::new(0, 0, 0, 0);

    /// Create padding from its left, top, right, and bottom amounts.
    pub const fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}

/// An axis-aligned rectangle with `u32` coordinates, stored as two corners `(x0, y0)` and
/// `(x1, y1)`.
///
/// `(x0, y0)` is the top-left (minimum) corner and `(x1, y1)` is the bottom-right (maximum) corner.
/// The rectangle is considered to be empty when `x0 >= x1` or `y0 >= y1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RectU32 {
    /// The minimum x coordinate (left edge).
    pub x0: u32,
    /// The minimum y coordinate (top edge).
    pub y0: u32,
    /// The maximum x coordinate (right edge, exclusive).
    pub x1: u32,
    /// The maximum y coordinate (bottom edge, exclusive).
    pub y1: u32,
}

impl RectU32 {
    /// Return the outward-rounded bounds in tile units.
    ///
    /// Pixel endpoints are converted without narrowing to the physical image domain.
    #[inline(always)]
    pub const fn to_tile_bounds(self) -> Self {
        let x0 = self.x0 / Tile::WIDTH_U32;
        let y0 = self.y0 / Tile::HEIGHT_U32;
        if self.is_empty() {
            return Self::new(x0, y0, x0, y0);
        }
        Self::new(
            x0,
            y0,
            self.x1.div_ceil(Tile::WIDTH_U32),
            self.y1.div_ceil(Tile::HEIGHT_U32),
        )
    }

    /// Convert tile bounds to pixels, rejecting an unrepresentable endpoint.
    #[inline(always)]
    pub const fn from_tile_bounds(tiles: Self) -> Self {
        Self::new(
            tiles
                .x0
                .checked_mul(Tile::WIDTH_U32)
                .expect("tile coordinate overflow"),
            tiles
                .y0
                .checked_mul(Tile::HEIGHT_U32)
                .expect("tile coordinate overflow"),
            tiles
                .x1
                .checked_mul(Tile::WIDTH_U32)
                .expect("tile coordinate overflow"),
            tiles
                .y1
                .checked_mul(Tile::HEIGHT_U32)
                .expect("tile coordinate overflow"),
        )
    }

    /// A rectangle with all coordinates set to zero.
    pub const ZERO: Self = Self {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    /// An empty, maximally inverted rectangle, useful as a starting value for incremental union
    /// operations.
    ///
    /// Has `(x0, y0) = (u32::MAX, u32::MAX)` and `(x1, y1) = (0, 0)`.
    pub const INVERTED: Self = Self {
        x0: u32::MAX,
        y0: u32::MAX,
        x1: 0,
        y1: 0,
    };

    /// Create a new rectangle from its corner coordinates.
    #[inline(always)]
    pub const fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The width of the rectangle (`x1 - x0`), saturating at zero.
    #[inline(always)]
    pub const fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// The height of the rectangle (`y1 - y0`), saturating at zero.
    #[inline(always)]
    pub const fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }

    /// Returns `true` if the rectangle has zero area (`x0 >= x1` or `y0 >= y1`).
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    /// Check if a point `(x, y)` is contained within this rectangle.
    ///
    /// Returns `true` if `x0 <= x < x1` and `y0 <= y < y1`.
    #[inline(always)]
    pub const fn contains(self, x: u32, y: u32) -> bool {
        (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)
    }

    /// Compute the intersection of two rectangles.
    ///
    /// The result may have zero area if the rectangles do not overlap, but is never inverted.
    #[inline(always)]
    pub const fn intersect(self, other: Self) -> Self {
        let x0 = const_max_u32(self.x0, other.x0);
        let y0 = const_max_u32(self.y0, other.y0);
        let x1 = const_min_u32(self.x1, other.x1);
        let y1 = const_min_u32(self.y1, other.y1);

        Self::new(x0, y0, const_max_u32(x1, x0), const_max_u32(y1, y0))
    }

    /// Expand this rectangle by the given left, top, right, and bottom padding.
    #[inline(always)]
    pub const fn expand(self, padding: PaddingU32) -> Self {
        Self {
            x0: self.x0.saturating_sub(padding.left),
            y0: self.y0.saturating_sub(padding.top),
            x1: self
                .x1
                .checked_add(padding.right)
                .expect("source rectangle overflow"),
            y1: self
                .y1
                .checked_add(padding.bottom)
                .expect("source rectangle overflow"),
        }
    }

    /// Return this rectangle relative to `origin`, clamping negative coordinates to zero.
    #[inline(always)]
    pub fn relative_to_origin(self, origin: (u32, u32)) -> Self {
        self.shift((-(origin.0 as i64), -(origin.1 as i64)))
    }

    /// Return a shifted version of the rectangle, clamping negative coordinates to zero.
    #[inline]
    pub fn shift(self, shift: (i64, i64)) -> Self {
        let shifted = |value: u32, delta: i64| {
            let value = i64::from(value)
                .checked_add(delta)
                .expect("source rectangle shift overflow");
            u32::try_from(value.max(0))
                .expect("source rectangle shift exceeds u32 coordinate domain")
        };
        Self::new(
            shifted(self.x0, shift.0),
            shifted(self.y0, shift.1),
            shifted(self.x1, shift.0),
            shifted(self.y1, shift.1),
        )
    }

    /// Expand this rectangle to also cover `other` (union in place).
    ///
    /// The union of `self` with a [`Self::INVERTED`] returns `self`.
    #[inline(always)]
    pub const fn union(&mut self, other: Self) {
        self.x0 = const_min_u32(self.x0, other.x0);
        self.y0 = const_min_u32(self.y0, other.y0);
        self.x1 = const_max_u32(self.x1, other.x1);
        self.y1 = const_max_u32(self.y1, other.y1);
    }

    /// Return the rect as a [`Rect`].
    pub fn as_rect(self) -> Rect {
        Rect::new(
            self.x0 as f64,
            self.y0 as f64,
            self.x1 as f64,
            self.y1 as f64,
        )
    }
}

impl From<RectU32> for SizeU32 {
    fn from(rect: RectU32) -> Self {
        Self::from_wh(rect.width(), rect.height())
    }
}

#[inline(always)]
const fn const_max_u32(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

#[inline(always)]
const fn const_min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

impl From<SizeU16> for SizeU32 {
    fn from(size: SizeU16) -> Self {
        Self::from_wh(u32::from(size.width()), u32::from(size.height()))
    }
}
impl From<RectU16> for RectU32 {
    fn from(rect: RectU16) -> Self {
        Self::new(
            rect.x0.into(),
            rect.y0.into(),
            rect.x1.into(),
            rect.y1.into(),
        )
    }
}
impl TryFrom<SizeU32> for SizeU16 {
    type Error = core::num::TryFromIntError;
    fn try_from(size: SizeU32) -> Result<Self, Self::Error> {
        Ok(Self::from_wh(
            size.width().try_into()?,
            size.height().try_into()?,
        ))
    }
}
impl TryFrom<RectU32> for RectU16 {
    type Error = core::num::TryFromIntError;
    fn try_from(rect: RectU32) -> Result<Self, Self::Error> {
        Ok(Self::new(
            rect.x0.try_into()?,
            rect.y0.try_into()?,
            rect.x1.try_into()?,
            rect.y1.try_into()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::RectU16;

    #[test]
    fn tile_bounds_preserve_last_physical_pixels() {
        for edge in 65532..=u16::MAX {
            let physical = RectU16::new(0, 0, edge, edge);
            let tiles = physical.to_tile_bounds();
            assert_eq!(u32::from(tiles.x1) * 4, u32::from(edge).next_multiple_of(4));
            let cover = RectU16::from_tile_bounds(tiles);
            assert_eq!(cover.intersect(physical), physical);
            assert!(cover.contains(edge - 1, edge - 1));
        }
    }

    #[test]
    fn rect_u16_relative_to_origin() {
        let rect = RectU16::new(10, 20, 30, 40);

        assert_eq!(rect.relative_to_origin((5, 12)), RectU16::new(5, 8, 25, 28));
    }

    #[test]
    fn rect_u16_relative_to_origin_clamps_to_zero() {
        let rect = RectU16::new(10, 20, 30, 40);

        assert_eq!(rect.relative_to_origin((20, 35)), RectU16::new(0, 0, 10, 5));
    }

    #[test]
    fn disjoint_intersection_is_empty_but_not_inverted() {
        let intersection = RectU16::new(0, 0, 4, 4).intersect(RectU16::new(8, 1, 12, 3));

        assert_eq!(intersection, RectU16::new(8, 1, 8, 3));
        assert!(intersection.is_empty());
        assert!(intersection.x0 <= intersection.x1);
        assert!(intersection.y0 <= intersection.y1);
    }
}

#[cfg(test)]
mod wide_tests {
    use super::{PaddingU32, RectU16, RectU32, SizeU16, SizeU32};
    use crate::util::RectExt;

    #[test]
    fn wide_tile_bounds_preserve_source_beyond_physical_image() {
        let rect = RectU32::new(65534, 70001, 65542, 70007);
        assert_eq!(
            rect.snap_to_tile_coordinates(),
            RectU32::new(65532, 70000, 65544, 70008)
        );
        assert_eq!(rect.shift((-65532, -70000)), RectU32::new(2, 1, 10, 7));
        assert_eq!(
            rect.expand(PaddingU32::new(2, 1, 2, 1)),
            RectU32::new(65532, 70000, 65544, 70008)
        );
        assert!(RectU16::try_from(rect).is_err());
        assert_eq!(
            RectU16::try_from(rect.shift((-65532, -70000))).unwrap(),
            RectU16::new(2, 1, 10, 7)
        );
        assert!(SizeU16::try_from(SizeU32::new(70000)).is_err());
    }

    #[test]
    #[should_panic(expected = "source rectangle shift exceeds u32 coordinate domain")]
    fn positive_shift_overflow_is_not_saturated() {
        RectU32::new(1, 1, u32::MAX, 4).shift((1, 0));
    }

    #[test]
    #[should_panic(expected = "tile coordinate overflow")]
    fn unrepresentable_tile_rounding_is_not_saturated() {
        RectU32::new(0, 0, u32::MAX, 1).snap_to_tile_coordinates();
    }
}
