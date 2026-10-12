// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter pixmaps placed in a source space that can exceed the `u16` domain.

use crate::filter::pixmap::FilterPixmap;
use vello_common::geometry::{PaddingU32, RectU16};
use vello_common::peniko::color::PremulRgba8;
use vello_common::tile::Tile;

/// A rectangle in a filter's source space. Empty rectangles are normalized to
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

    pub(crate) fn new(x0: i64, y0: i64, x1: i64, y1: i64) -> Self {
        if x0 >= x1 || y0 >= y1 {
            return Self::EMPTY;
        }
        Self { x0, y0, x1, y1 }
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

    pub(crate) fn translate(self, (dx, dy): (i64, i64)) -> Self {
        if self.is_empty() {
            return self;
        }
        Self::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }

    pub(crate) fn expand(self, padding: PaddingU32) -> Self {
        if self.is_empty() {
            return self;
        }
        Self::new(
            self.x0 - i64::from(padding.left),
            self.y0 - i64::from(padding.top),
            self.x1 + i64::from(padding.right),
            self.y1 + i64::from(padding.bottom),
        )
    }

    /// Shrink by `padding`, the inverse of [`Bounds::expand`].
    pub(crate) fn shrink(self, padding: PaddingU32) -> Self {
        Self::new(
            self.x0 + i64::from(padding.left),
            self.y0 + i64::from(padding.top),
            self.x1 - i64::from(padding.right),
            self.y1 - i64::from(padding.bottom),
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

    fn index(&self, x: i64, y: i64) -> usize {
        let bounds = self.bounds;
        usize::try_from((y - bounds.y0) * bounds.width() + x - bounds.x0).unwrap()
    }
}
