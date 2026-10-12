// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! How the visible output of a filter layer is computed from the parts of its
//! source it reads.
//!
//! Each region of the output reads the source around itself: widened by the
//! reach of a blur, or moved back by the offset of an offset or drop shadow,
//! whose output is a copy of the source moved by whole pixels, blurred and
//! recolored first for a drop shadow, under the unmoved source for a drop
//! shadow that keeps it. Only those parts of the source are rendered, and each
//! is filtered in a pixmap around its own content, so the work follows the
//! visible output, not the extent of the source or the distance it moves.
//!
//! The reach of a blur, and with it the source it reads, grows with its
//! standard deviation. A blur decimates its source in steps of two before it
//! convolves it, so a blur that decimates more than [`EXACT_DECIMATIONS`] times
//! reads its source rendered that much coarser, and only its output is
//! upscaled back to full resolution. Its work then stays bounded however wide
//! the blur is. Blurs that decimate less read the source at full resolution
//! and give exactly the result of filtering the whole source.

use crate::filter::context::ScratchBuffer;
use crate::filter::drop_shadow::source_over;
use crate::filter::filter_reduced;
use crate::filter::pixmap::FilterPixmap;
use crate::filter::placed::{Bounds, PlacedPixmap, Regions};
use alloc::vec::Vec;
use vello_common::filter::{Displacement, FilterData, PreparedFilter};
use vello_common::geometry::PaddingU32;

/// The number of decimations a blur performs on a source rendered at full
/// resolution. A blur that decimates more reads its source rendered so much
/// coarser that this many decimations remain.
pub(crate) const EXACT_DECIMATIONS: usize = 5;

/// How a filter layer reads its source and writes its output, in the pixel
/// grid of the output.
#[derive(Debug)]
pub(crate) struct FilterPlan {
    /// The filter, with its parameters in the pixel grid of the output.
    filter: FilterData,
    effect: Effect,
    /// By how many steps of two the source the filter reads is coarser than the output.
    reduction: u32,
    outputs: Vec<Output>,
}

#[derive(Clone, Copy, Debug)]
enum Effect {
    /// Filtered as a whole in a pixmap around the source it reads.
    Whole {
        source_padding: PaddingU32,
        filter_padding: PaddingU32,
    },
    /// Moved, see the module documentation.
    Displaced(Displacement),
}

/// A region of the output with the source it reads.
#[derive(Debug)]
struct Output {
    bounds: Bounds,
    /// The source composited unmoved, in the output's grid.
    kept: Bounds,
    /// The source that is filtered, in the grid of the source the filter reads.
    filtered: Bounds,
}

/// Where the parts of one output region are computed, given where the source has content.
#[derive(Debug)]
struct Layout {
    /// The pixmap the filter runs on, in the grid of the source it reads.
    filtered: Bounds,
    /// The filtered output, in the output's grid before it is moved.
    result: Bounds,
    /// The kept source, in the output's grid.
    kept: Bounds,
    /// Where the output has content.
    extent: Bounds,
}

impl FilterPlan {
    /// Plan the output of `filter` inside `outputs`. Its parameters are in the
    /// pixel grid of the output.
    pub(crate) fn new(filter: FilterData, outputs: &Regions) -> Self {
        let decimations = match PreparedFilter::new(&filter.filter, &filter.transform) {
            PreparedFilter::GaussianBlur(blur) => blur.n_decimations,
            PreparedFilter::DropShadow(shadow) => shadow.n_decimations,
            PreparedFilter::Flood(_) | PreparedFilter::Offset(_) => 0,
        };
        let reduction = u32::try_from(decimations.saturating_sub(EXACT_DECIMATIONS)).unwrap();
        let effect = match filter.displacement() {
            Some(displacement) => Effect::Displaced(displacement),
            None => Effect::Whole {
                source_padding: filter.source_padding,
                filter_padding: filter.filter_padding,
            },
        };
        let outputs = outputs
            .iter()
            .map(|bounds| {
                let (kept, filtered) = match effect {
                    Effect::Whole { source_padding, .. } => {
                        (Bounds::EMPTY, bounds.expand(source_padding))
                    }
                    Effect::Displaced(displacement) => {
                        let (dx, dy) = displacement.offset;
                        let moved = bounds
                            .translate((-dx, -dy))
                            .expand(displacement.blur_padding);
                        let kept = if displacement.keeps_source {
                            bounds
                        } else {
                            Bounds::EMPTY
                        };
                        (kept, moved)
                    }
                };
                Output {
                    bounds,
                    kept,
                    filtered: filtered.coarsen(reduction),
                }
            })
            .collect();
        Self {
            filter,
            effect,
            reduction,
            outputs,
        }
    }

    /// Return by how many steps of two the source the filter reads is coarser
    /// than its output.
    pub(crate) fn reduction(&self) -> u32 {
        self.reduction
    }

    /// Return the regions of the source to render: those the output keeps, in
    /// the output's grid, and those the filter reads, [`Self::reduction`] steps
    /// coarser.
    pub(crate) fn source_regions(&self) -> (Regions, Regions) {
        let mut kept = Regions::default();
        let mut filtered = Regions::default();
        for output in &self.outputs {
            kept.insert(output.kept);
            filtered.insert(output.filtered);
        }
        (kept, filtered)
    }

    /// Return where the output has content, given the bounds of the rendered
    /// source as in [`Self::render`].
    pub(crate) fn extents(&self, kept: &[Bounds], filtered: &[Bounds]) -> Vec<Bounds> {
        self.outputs
            .iter()
            .map(|output| self.layout(output, kept, filtered).extent)
            .collect()
    }

    /// Compute the output from the rendered source: `kept` covers the kept
    /// regions and `filtered` the regions the filter reads, see
    /// [`Self::source_regions`]. `apply` applies a prepared filter to a pixmap
    /// in the output's grid. The pixmaps cover the extents of the output.
    pub(crate) fn render(
        &self,
        kept: &[PlacedPixmap],
        filtered: &[PlacedPixmap],
        scratch: &mut ScratchBuffer,
        apply: &mut dyn FnMut(&mut FilterPixmap, PreparedFilter, &mut ScratchBuffer),
    ) -> Vec<PlacedPixmap> {
        let bounds = |pieces: &[PlacedPixmap]| -> Vec<Bounds> {
            pieces.iter().map(|piece| piece.bounds).collect()
        };
        let (kept_bounds, filtered_bounds) = (bounds(kept), bounds(filtered));
        self.outputs
            .iter()
            .map(|output| {
                let layout = self.layout(output, &kept_bounds, &filtered_bounds);
                let mut pixmap = PlacedPixmap::new(layout.extent);
                if !layout.filtered.is_empty() {
                    let result = self.filter_source(&layout, filtered, scratch, apply);
                    pixmap.copy_from(&result.translate(self.offset()), layout.extent);
                }
                if !layout.kept.is_empty() {
                    let kept = assemble(layout.kept, kept);
                    pixmap.combine(&kept, layout.kept, source_over);
                }
                pixmap
            })
            .collect()
    }

    /// Return the pixmaps the filter runs on, given the bounds of the rendered
    /// source as in [`Self::render`].
    #[cfg(test)]
    pub(crate) fn pixmaps(&self, kept: &[Bounds], filtered: &[Bounds]) -> Vec<Bounds> {
        self.outputs
            .iter()
            .map(|output| self.layout(output, kept, filtered).filtered)
            .collect()
    }

    /// Return by how much the filtered source is moved.
    fn offset(&self) -> (i64, i64) {
        match self.effect {
            Effect::Whole { .. } => (0, 0),
            Effect::Displaced(displacement) => displacement.offset,
        }
    }

    /// Lay out `output`, given the bounds of the rendered pieces of the kept
    /// source and of the source the filter reads, which are tight around their
    /// content.
    fn layout(&self, output: &Output, kept: &[Bounds], filtered: &[Bounds]) -> Layout {
        let (dx, dy) = self.offset();
        let padding = match self.effect {
            Effect::Whole { filter_padding, .. } => filter_padding,
            Effect::Displaced(displacement) => displacement.blur_padding,
        };
        let source = content(filtered, output.filtered);
        let mut pixmap = Bounds::EMPTY;
        let mut result = Bounds::EMPTY;
        if !source.is_empty() {
            // The pixmap's origin anchors the blur's decimation, so it follows
            // the content wherever the source it reads holds all of it.
            pixmap = source
                .expand(coarsen(padding, self.reduction))
                .snap_to_tiles();
            result = pixmap
                .refine(self.reduction)
                .intersect(output.bounds.translate((-dx, -dy)));
            if result.is_empty() {
                pixmap = Bounds::EMPTY;
            }
        }
        let kept = content(kept, output.kept);
        Layout {
            filtered: pixmap,
            result,
            kept,
            extent: result.translate((dx, dy)).union(kept),
        }
    }

    /// Filter the source in `filtered` inside the pixmap of `layout` and return
    /// the result.
    fn filter_source(
        &self,
        layout: &Layout,
        filtered: &[PlacedPixmap],
        scratch: &mut ScratchBuffer,
        apply: &mut dyn FnMut(&mut FilterPixmap, PreparedFilter, &mut ScratchBuffer),
    ) -> PlacedPixmap {
        let mut pixmap = assemble(layout.filtered, filtered);
        let prepared = PreparedFilter::new(&self.filter.filter, &self.filter.transform);
        if self.reduction == 0 {
            apply(&mut pixmap.pixmap, prepared, scratch);
            return pixmap.crop(layout.result);
        }
        let origin = pixmap.bounds.refine(self.reduction);
        PlacedPixmap {
            pixmap: filter_reduced(
                prepared,
                pixmap.pixmap,
                self.reduction,
                layout.result.translate((-origin.x0, -origin.y0)),
                scratch,
            ),
            bounds: layout.result,
        }
    }
}

/// Return the bounds of the content inside `area`, given the bounds of the
/// rendered pieces of a source, which are tight around their content.
fn content(pieces: &[Bounds], area: Bounds) -> Bounds {
    pieces.iter().fold(Bounds::EMPTY, |content, piece| {
        content.union(piece.intersect(area))
    })
}

/// Copy the pixels of `pieces` inside `bounds` into a new pixmap.
fn assemble(bounds: Bounds, pieces: &[PlacedPixmap]) -> PlacedPixmap {
    let mut pixmap = PlacedPixmap::new(bounds);
    for piece in pieces {
        pixmap.copy_from(piece, bounds);
    }
    pixmap
}

/// Return the padding that covers `padding` in a grid `levels` steps of two coarser.
fn coarsen(padding: PaddingU32, levels: u32) -> PaddingU32 {
    let side = |value: u32| value.div_ceil(1 << levels);
    PaddingU32::new(
        side(padding.left),
        side(padding.top),
        side(padding.right),
        side(padding.bottom),
    )
}
