// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Offset and drop shadow layers, assembled from the parts of their source
//! that the visible output reads.
//!
//! The output of such a layer is its source moved by whole pixels, blurred and
//! recolored first for a drop shadow, with the unmoved source over it for a
//! drop shadow that keeps it. Filtering one pixmap that spans the source and
//! its moved copy would allocate and process the whole distance between them.
//! Instead, the moved part is filtered in a pixmap around its own source and
//! placed at its destination, so the work grows with the view and the blur,
//! not with the offset.

use crate::filter::drop_shadow::source_over;
use crate::filter::pixmap::FilterPixmap;
use crate::filter::placed::{Bounds, PlacedPixmap};
use vello_common::filter::Displacement;

/// Where an offset or drop shadow layer reads its source and writes its
/// output, in the layer's source space.
#[derive(Debug)]
pub(crate) struct DisplacedPlan {
    offset: (i64, i64),
    /// Source composited unmoved.
    kept: Bounds,
    /// Source the moved copy reads.
    moved: Bounds,
    /// Pixmap the moved copy is filtered in, before it is moved.
    filtered: Bounds,
    /// Output, inside the view.
    output: Bounds,
}

impl DisplacedPlan {
    /// Plan the output inside `view` of a layer whose content lies in `content`.
    pub(crate) fn new(displacement: Displacement, content: Bounds, view: Bounds) -> Self {
        let [kept_region, moved_region] = source_regions(displacement, view);
        let kept = content.intersect(kept_region);
        let moved = content.intersect(moved_region);
        // Like a blur layer, the moved copy is filtered with room for its blur,
        // starting on a tile so that its decimation phase follows the content.
        let filtered = moved.expand(displacement.blur_padding).snap_to_tiles();
        let moved_output = filtered.translate(displacement.offset).intersect(view);
        Self {
            offset: displacement.offset,
            kept,
            moved,
            filtered,
            output: kept.union(moved_output).snap_to_tiles().intersect(view),
        }
    }

    /// Return the output bounds, which start on a tile if `view` does.
    pub(crate) fn output(&self) -> Bounds {
        self.output
    }

    /// Return the tile-aligned areas of the source to render for
    /// [`DisplacedPlan::assemble`], without overlap.
    pub(crate) fn source_areas(&self) -> impl Iterator<Item = Bounds> {
        let kept = self.kept.snap_to_tiles();
        let moved = self.moved.snap_to_tiles();
        let areas = if kept.intersect(moved).is_empty() {
            [kept, moved]
        } else {
            [kept.union(moved), Bounds::EMPTY]
        };
        areas.into_iter().filter(|area| !area.is_empty())
    }

    /// Assemble the output from `pieces` of the rendered source, which cover
    /// the source areas. `filter` blurs and recolors the moved copy in place.
    pub(crate) fn assemble(
        &self,
        pieces: impl IntoIterator<Item = PlacedPixmap>,
        filter: impl FnOnce(&mut FilterPixmap),
    ) -> PlacedPixmap {
        let mut kept = PlacedPixmap::new(self.kept);
        let mut moved = PlacedPixmap::new(self.filtered);
        for piece in pieces {
            kept.copy_from(&piece, self.kept);
            moved.copy_from(&piece, self.moved);
        }
        let mut output = PlacedPixmap::new(self.output);
        if !self.moved.is_empty() {
            filter(&mut moved.pixmap);
            output.copy_from(&moved.translate(self.offset), self.output);
        }
        output.combine(&kept, self.kept, source_over);
        output
    }
}

/// Return the regions of the source that the output inside `view` reads: the
/// part kept unmoved, and the part that the moved copy reads.
pub(crate) fn source_regions(displacement: Displacement, view: Bounds) -> [Bounds; 2] {
    let (dx, dy) = displacement.offset;
    let kept = if displacement.keeps_source {
        view
    } else {
        Bounds::EMPTY
    };
    let moved = view.translate((-dx, -dy)).expand(displacement.blur_padding);
    [kept, moved]
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::geometry::PaddingU32;

    fn shadow(offset: (i64, i64), blur: u32, keeps_source: bool) -> Displacement {
        Displacement {
            offset,
            blur_padding: PaddingU32::new(blur, blur, blur, blur),
            keeps_source,
        }
    }

    fn area(bounds: Bounds) -> i64 {
        bounds.width() * bounds.height()
    }

    #[test]
    fn allocation_does_not_grow_with_the_offset() {
        let view = Bounds::new(0, 0, 12, 12);
        for distance in [16_i64, 65_536, 1 << 24, 1 << 31] {
            for (dx, dy) in [(distance, distance), (-distance, 3), (5, -distance)] {
                // A source at the moved region and one inside the view, so both parts are visible.
                let far = Bounds::new(4 - dx, 4 - dy, 8 - dx, 8 - dy);
                let content = far.union(Bounds::new(-8, -8, 4, 4));
                let plan = DisplacedPlan::new(shadow((dx, dy), 8, true), content, view);
                let allocated = area(plan.kept) + area(plan.filtered) + area(plan.output);
                let rendered: i64 = plan.source_areas().map(area).sum();
                assert!(allocated <= 3 * 28 * 28, "{dx}, {dy}: {plan:?}");
                assert!(rendered <= 2 * 28 * 28, "{dx}, {dy}: {plan:?}");
                assert_eq!(plan.output, view, "{dx}, {dy}");
            }
        }
    }

    #[test]
    fn regions_cover_kept_source_and_blur_support_of_moved_copy() {
        let view = Bounds::new(100, 200, 112, 212);
        assert_eq!(
            source_regions(shadow((-1000, 7), 8, true), view),
            [view, Bounds::new(1092, 185, 1120, 213)]
        );
        assert_eq!(
            source_regions(shadow((3, 3), 0, false), view),
            [Bounds::EMPTY, Bounds::new(97, 197, 109, 209)]
        );
    }
}
