// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Spans for delimiting horizontal regions.

use vello_common::tile::Tile;

/// A horizontal span in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct Span {
    /// The horizontal start position in pixels.
    x: u32,
    /// The horizontal span width in pixels.
    width: u32,
}

impl Span {
    /// Creates a span from pixel coordinates.
    pub fn new(x: u32, width: u32) -> Self {
        Self { x, width }
    }

    /// Expands this span to the tiles covering it. Empty spans remain empty.
    pub fn tile_aligned(self) -> TileAlignedSpan {
        let start = self.tile_x();
        let count = if self.width == 0 {
            0
        } else {
            self.tile_end() - start
        };

        TileAlignedSpan::from_tiles(start, count)
    }

    /// Returns the horizontal start position in tile coordinates.
    pub fn tile_x(self) -> u32 {
        self.x / Tile::WIDTH_U32
    }

    /// Returns the exclusive horizontal end position in tile coordinates.
    pub fn tile_end(self) -> u32 {
        self.pixel_end().div_ceil(Tile::WIDTH_U32)
    }

    /// Extends this span to include another span.
    pub fn extend(&mut self, other: Self) {
        let x = self.x.min(other.x);
        let end = self.pixel_end().max(other.pixel_end());
        *self = Self::new(x, end - x);
    }

    /// Returns the intersection of this span with another span.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let end = self.pixel_end().min(other.pixel_end());
        (x < end).then(|| Self::new(x, end - x))
    }

    /// Returns the horizontal start position in pixels.
    pub fn pixel_x(self) -> u32 {
        self.x
    }

    /// Returns the horizontal span width in pixels.
    pub fn pixel_width(self) -> u32 {
        self.width
    }

    /// Returns the exclusive horizontal end position in pixels.
    pub fn pixel_end(self) -> u32 {
        self.pixel_x()
            .checked_add(self.pixel_width())
            .expect("pixel span end overflow")
    }
}

/// A horizontal pixel span aligned to tile coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct TileAlignedSpan {
    start: u32,
    end: u32,
}

impl TileAlignedSpan {
    /// Creates a tile-aligned span from a tile start and tile count.
    pub fn from_tiles(start: u32, count: u32) -> Self {
        let end = start.checked_add(count).expect("tile span end overflow");
        assert!(
            end <= u32::MAX.div_ceil(Tile::WIDTH_U32),
            "tile span exceeds coordinate domain"
        );
        Self { start, end }
    }

    /// Intersects with physical pixels before narrowing the padded tile endpoint.
    pub fn intersect_pixels(self, bounds: Span) -> Option<Span> {
        let start = self.pixel_x().max(bounds.pixel_x() as usize);
        let end = self.pixel_end().min(bounds.pixel_end() as usize);
        (start < end).then(|| {
            Span::new(
                u32::try_from(start).unwrap(),
                u32::try_from(end - start).unwrap(),
            )
        })
    }

    /// Returns the horizontal start position in pixels.
    pub fn pixel_x(self) -> usize {
        (self.start as usize)
            .checked_mul(Tile::WIDTH_U32 as usize)
            .expect("tile start exceeds address space")
    }

    /// Returns the horizontal width in pixels, including scratch padding.
    pub fn pixel_width(self) -> usize {
        ((self.end - self.start) as usize)
            .checked_mul(Tile::WIDTH_U32 as usize)
            .expect("tile width exceeds address space")
    }

    /// Returns the exclusive horizontal end in pixels, including scratch padding.
    pub fn pixel_end(self) -> usize {
        (self.end as usize)
            .checked_mul(Tile::WIDTH_U32 as usize)
            .expect("tile end exceeds address space")
    }

    /// Returns the horizontal start in tile coordinates.
    pub fn tile_x(self) -> u32 {
        self.start
    }

    /// Returns the exclusive horizontal end in tile coordinates.
    pub fn tile_end(self) -> u32 {
        self.end
    }

    /// Extends this span to cover another tile-aligned span.
    pub fn extend(&mut self, other: Self) {
        self.start = self.start.min(other.start);
        self.end = self.end.max(other.end);
    }

    /// Intersects two tile-aligned spans.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (start < end).then_some(Self { start, end })
    }
}

impl TryFrom<Span> for TileAlignedSpan {
    type Error = &'static str;

    fn try_from(span: Span) -> Result<Self, Self::Error> {
        if !span.x.is_multiple_of(Tile::WIDTH_U32) || !span.width.is_multiple_of(Tile::WIDTH_U32) {
            return Err("span is not tile-aligned");
        }

        if span.x.checked_add(span.width).is_none() {
            return Err("tile span end overflow");
        }

        Ok(Self::from_tiles(
            span.x / Tile::WIDTH_U32,
            span.width / Tile::WIDTH_U32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::span::{Span, TileAlignedSpan};
    use vello_common::tile::Tile;

    #[test]
    fn tile_aligned_rounds_outward() {
        let tile = Tile::WIDTH_U32;
        for (span, expected) in [
            (Span::new(tile, tile), Span::new(tile, tile)),
            (Span::new(tile + 1, tile - 1), Span::new(tile, tile)),
            (Span::new(tile, tile + 1), Span::new(tile, 2 * tile)),
            (Span::new(tile + 1, tile), Span::new(tile, 2 * tile)),
        ] {
            assert_eq!(
                span.tile_aligned()
                    .intersect_pixels(Span::new(0, u32::from(u16::MAX))),
                Some(expected),
                "{span:?}"
            );
        }
    }

    #[test]
    fn tile_aligned_keeps_empty_spans_empty() {
        for x in [0, Tile::WIDTH_U32, Tile::WIDTH_U32 + 1] {
            let aligned = Span::new(x, 0).tile_aligned();
            assert_eq!(aligned.pixel_width(), 0, "x = {x}");
        }
    }

    #[test]
    fn try_from_preserves_aligned_bounds() {
        let max_aligned = u32::MAX / Tile::WIDTH_U32 * Tile::WIDTH_U32;
        for span in [
            Span::new(Tile::WIDTH_U32, 3 * Tile::WIDTH_U32),
            Span::new(0, max_aligned),
            Span::new(max_aligned, 0),
        ] {
            let aligned = TileAlignedSpan::try_from(span).unwrap();
            assert_eq!(aligned.pixel_x(), (span.pixel_x() as usize));
            assert_eq!(aligned.pixel_width(), (span.pixel_width() as usize));
        }
    }

    #[test]
    fn padded_terminal_tile_preserves_physical_pixels() {
        assert_eq!(size_of::<TileAlignedSpan>(), 8);
        for width in 65532..=u32::from(u16::MAX) {
            let physical = Span::new(0, width);
            let aligned = physical.tile_aligned();
            let expected_end = (width as usize).div_ceil(4) * 4;
            assert_eq!(aligned.pixel_end(), expected_end);
            assert_eq!(aligned.pixel_width(), expected_end);
            assert_eq!(aligned.intersect_pixels(physical), Some(physical));
        }
        let tail = TileAlignedSpan::from_tiles(16383, 1);
        assert_eq!(tail.pixel_end(), 65536);
        assert_eq!(
            tail.intersect_pixels(Span::new(0, u32::from(u16::MAX))),
            Some(Span::new(65532, 3))
        );
        let empty = TileAlignedSpan::from_tiles(16384, 0);
        assert_eq!(empty.pixel_x(), 65536);
        assert_eq!(empty.pixel_width(), 0);
        assert_eq!(
            empty.intersect_pixels(Span::new(0, u32::from(u16::MAX))),
            None
        );
        assert_eq!(tail.intersect(empty), None);
        let mut union = TileAlignedSpan::from_tiles(0, 1);
        union.extend(tail);
        assert_eq!(union, TileAlignedSpan::from_tiles(0, 16384));
        assert_eq!(union.intersect(tail), Some(tail));
    }

    #[test]
    fn try_from_rejects_unaligned_bounds() {
        for span in [
            Span::new(Tile::WIDTH_U32 + 1, Tile::WIDTH_U32),
            Span::new(Tile::WIDTH_U32, Tile::WIDTH_U32 + 1),
            Span::new(Tile::WIDTH_U32 + 1, 0),
        ] {
            assert!(TileAlignedSpan::try_from(span).is_err(), "{span:?}");
        }
    }
}
