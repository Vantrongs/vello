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
use crate::filter::placed::{Bounds, Index, Pieces, PlacedPixmap, Regions, Shift, pixel_size};
use crate::kurbo::Affine;
use alloc::vec::Vec;
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::geometry::PaddingU64;

/// The number of decimations a blur performs on a source rendered at full
/// resolution. A blur that decimates more reads its source rendered so much
/// coarser that this many decimations remain.
pub(crate) const EXACT_DECIMATIONS: usize = 5;

/// Return `filter` with its parameters in the grid of root target pixels
/// `2^level` wide.
pub(crate) fn filter_at(filter: &FilterData, level: u32) -> FilterData {
    FilterData::new(
        filter.filter.clone(),
        Affine::scale(1.0 / pixel_size(level)) * filter.transform,
    )
}

/// How a filter reads its source and writes its output in one pixel grid.
#[derive(Clone, Debug)]
pub(crate) struct Reach {
    /// The filter, with its parameters in the pixel grid of the output.
    filter: FilterData,
    effect: Effect,
    /// By how many steps of two the source the filter reads is coarser than the output.
    reduction: u32,
}

#[derive(Clone, Copy, Debug)]
enum Effect {
    /// Filtered as a whole in a pixmap around the source it reads, with
    /// paddings in the grid of that source, and moved by `offset` in the
    /// output's grid.
    Whole {
        source_padding: PaddingU64,
        filter_padding: PaddingU64,
        offset: (i64, i64),
    },
    /// Moved by `offset` in the output's grid, see the module documentation,
    /// after a blur that reads `blur_padding` around each pixel in the grid of
    /// the source it reads. The source it keeps is moved by `kept`.
    Displaced {
        offset: (i64, i64),
        kept: (i64, i64),
        blur_padding: PaddingU64,
        keeps_source: bool,
    },
}

/// The offset of `filter` in the grid of root target pixels `2^level` wide, if it moves its source.
fn offset_at(filter: &FilterData, level: u32) -> (Shift, Shift) {
    filter_at(filter, level)
        .displacement()
        .map_or_else(Default::default, |displacement| {
            let (dx, dy) = displacement.offset;
            (Shift::of(dx), Shift::of(dy))
        })
}

impl Reach {
    /// Return how `filter`, whose parameters are in the root grid, reads and
    /// writes in the grid of root target pixels `2^level` wide, with its output
    /// moved further by the offset filters `shifts`. Unless `moves` is set, the
    /// offset of `filter` itself is left out.
    pub(crate) fn new(filter: &FilterData, level: u32, shifts: &[FilterData], moves: bool) -> Self {
        let output = filter_at(filter, level);
        let decimations = match PreparedFilter::new(&output.filter, &output.transform) {
            PreparedFilter::GaussianBlur(blur) => blur.n_decimations,
            PreparedFilter::DropShadow(shadow) => shadow.n_decimations,
            PreparedFilter::Flood(_) | PreparedFilter::Offset(_) => 0,
        };
        let reduction = u32::try_from(decimations.saturating_sub(EXACT_DECIMATIONS)).unwrap();
        // Paddings are taken in the grid they apply in, where they stay small
        // however wide the blur is.
        let source = if reduction == 0 {
            output.clone()
        } else {
            filter_at(filter, level + reduction)
        };
        // Offsets are summed before they are narrowed, so that opposite ones
        // cancel however far they reach.
        let shift = shifts
            .iter()
            .map(|shift| offset_at(shift, level))
            .fold(<(Shift, Shift)>::default(), |(x, y), (dx, dy)| {
                (x.add(dx), y.add(dy))
            });
        let narrow = |(x, y): (Shift, Shift)| (x.saturated(), y.saturated());
        let effect = match (output.displacement(), source.displacement()) {
            (Some(displacement), Some(coarse)) => {
                let own = if moves {
                    offset_at(filter, level)
                } else {
                    Default::default()
                };
                Effect::Displaced {
                    offset: narrow((shift.0.add(own.0), shift.1.add(own.1))),
                    kept: narrow(shift),
                    blur_padding: coarse.blur_padding,
                    keeps_source: displacement.keeps_source,
                }
            }
            _ => Effect::Whole {
                source_padding: source.source_padding,
                filter_padding: source.filter_padding,
                offset: narrow(shift),
            },
        };
        Self {
            filter: output,
            effect,
            reduction,
        }
    }

    /// Return by how many steps of two the source the filter reads is coarser
    /// than its output.
    pub(crate) fn reduction(&self) -> u32 {
        self.reduction
    }

    /// Return the source that the output inside `output` reads: the part it
    /// keeps, in the output's grid, and the part the filter reads,
    /// [`Self::reduction`] steps coarser.
    pub(crate) fn reads(&self, output: Bounds) -> (Bounds, Bounds) {
        let (dx, dy) = self.offset();
        let moved = output.translate((-dx, -dy)).coarsen(self.reduction);
        match self.effect {
            Effect::Whole { source_padding, .. } => (Bounds::EMPTY, moved.expand(source_padding)),
            Effect::Displaced {
                kept: (kx, ky),
                blur_padding,
                keeps_source,
                ..
            } => {
                let kept = if keeps_source {
                    output.translate((-kx, -ky))
                } else {
                    Bounds::EMPTY
                };
                (kept, moved.expand(blur_padding))
            }
        }
    }

    /// Return whether the output holds a copy of the source moved only by
    /// [`Self::kept_offset`], where [`Self::reads`] places the part it keeps.
    pub(crate) fn keeps_source(&self) -> bool {
        matches!(
            self.effect,
            Effect::Displaced {
                keeps_source: true,
                ..
            }
        )
    }

    /// Return where the filtered part of the output can have content, given
    /// that the source the filter reads has content only inside `filtered`.
    pub(crate) fn writes(&self, filtered: Bounds) -> Bounds {
        filtered
            .expand(self.padding())
            .snap_to_tiles()
            .refine(self.reduction)
            .translate(self.offset())
    }

    /// Return by how much the filtered source is moved.
    fn offset(&self) -> (i64, i64) {
        match self.effect {
            Effect::Whole { offset, .. } | Effect::Displaced { offset, .. } => offset,
        }
    }

    /// Return by how much the kept source is moved.
    pub(crate) fn kept_offset(&self) -> (i64, i64) {
        match self.effect {
            Effect::Whole { .. } => (0, 0),
            Effect::Displaced { kept, .. } => kept,
        }
    }

    /// Return how far the pixmap a filter runs on reaches around its content,
    /// in the grid of the source it reads.
    fn padding(&self) -> PaddingU64 {
        match self.effect {
            Effect::Whole { filter_padding, .. } => filter_padding,
            Effect::Displaced { blur_padding, .. } => blur_padding,
        }
    }
}

/// How a filter layer reads its source and writes its output, in the pixel
/// grid of the output.
#[derive(Debug)]
pub(crate) struct FilterPlan {
    reach: Reach,
    outputs: Vec<Output>,
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
    /// The kept source, in the output's grid before it is moved.
    kept: Bounds,
    /// Where the output has content.
    extent: Bounds,
}

impl FilterPlan {
    /// Plan the output of a filter that reads and writes as `reach` inside `outputs`.
    pub(crate) fn new(reach: Reach, outputs: &Regions) -> Self {
        let outputs = outputs
            .iter()
            .map(|bounds| {
                let (kept, filtered) = reach.reads(bounds);
                Output {
                    bounds,
                    kept,
                    filtered,
                }
            })
            .collect();
        Self { reach, outputs }
    }

    /// Return by how many steps of two the source the filter reads is coarser
    /// than its output.
    pub(crate) fn reduction(&self) -> u32 {
        self.reach.reduction
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
    pub(crate) fn extents(&self, kept: &Index, filtered: &Index) -> Vec<Bounds> {
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
        kept: &Pieces,
        filtered: &Pieces,
        scratch: &mut ScratchBuffer,
        apply: &mut dyn FnMut(&mut FilterPixmap, PreparedFilter, &mut ScratchBuffer),
    ) -> Vec<PlacedPixmap> {
        self.outputs
            .iter()
            .map(|output| {
                let layout = self.layout(output, kept.index(), filtered.index());
                let mut pixmap = PlacedPixmap::new(layout.extent);
                if !layout.filtered.is_empty() {
                    let result = self.filter_source(&layout, filtered, scratch, apply);
                    pixmap.copy_from(&result.translate(self.reach.offset()), layout.extent);
                }
                if !layout.kept.is_empty() {
                    let kept = kept.crop(layout.kept).translate(self.reach.kept_offset());
                    pixmap.combine(&kept, kept.bounds, source_over);
                }
                pixmap
            })
            .collect()
    }

    /// Return the pixmaps the filter runs on, given the bounds of the rendered
    /// source as in [`Self::render`].
    #[cfg(test)]
    pub(crate) fn pixmaps(&self, kept: &Index, filtered: &Index) -> Vec<Bounds> {
        self.outputs
            .iter()
            .map(|output| self.layout(output, kept, filtered).filtered)
            .collect()
    }

    /// Lay out `output`, given the bounds of the rendered pieces of the kept
    /// source and of the source the filter reads, which are tight around their
    /// content.
    fn layout(&self, output: &Output, kept: &Index, filtered: &Index) -> Layout {
        let (dx, dy) = self.reach.offset();
        let source = content(filtered, output.filtered);
        let mut pixmap = Bounds::EMPTY;
        let mut result = Bounds::EMPTY;
        if !source.is_empty() {
            // The pixmap's origin anchors the blur's decimation, so it follows
            // the content wherever the source it reads holds all of it.
            pixmap = source.expand(self.reach.padding()).snap_to_tiles();
            result = pixmap
                .refine(self.reduction())
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
            extent: result
                .translate((dx, dy))
                .union(kept.translate(self.reach.kept_offset())),
        }
    }

    /// Filter the source in `filtered` inside the pixmap of `layout` and return
    /// the result.
    fn filter_source(
        &self,
        layout: &Layout,
        filtered: &Pieces,
        scratch: &mut ScratchBuffer,
        apply: &mut dyn FnMut(&mut FilterPixmap, PreparedFilter, &mut ScratchBuffer),
    ) -> PlacedPixmap {
        let mut pixmap = filtered.crop(layout.filtered);
        let filter = &self.reach.filter;
        let prepared = PreparedFilter::new(&filter.filter, &filter.transform);
        if self.reduction() == 0 {
            apply(&mut pixmap.pixmap, prepared, scratch);
            return pixmap.crop(layout.result);
        }
        let origin = (pixmap.bounds.x0, pixmap.bounds.y0);
        PlacedPixmap {
            pixmap: filter_reduced(
                prepared,
                pixmap.pixmap,
                self.reduction(),
                origin,
                layout.result,
                scratch,
            ),
            bounds: layout.result,
        }
    }
}

/// Return the bounds of the content inside `area`, given the bounds of the
/// rendered pieces of a source, which are tight around their content.
fn content(pieces: &Index, area: Bounds) -> Bounds {
    pieces
        .touching(area)
        .fold(Bounds::EMPTY, |content, (_, piece)| {
            content.union(piece.intersect(area))
        })
}
