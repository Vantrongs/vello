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
//! convolves it, so a blur that decimates more than [`EXACT_DECIMATIONS`]
//! times reads its source averaged over squares of pixels, so many steps
//! coarser that this many decimations remain, and only its output is upscaled
//! back to full resolution. Its memory then stays bounded however wide the
//! blur is, while every pixel of the source still counts.

use crate::filter::context::ScratchBuffer;
use crate::filter::drop_shadow::source_over;
use crate::filter::pixmap::FilterPixmap;
use crate::filter::placed::{
    Bounds, Coord, Index, Pieces, PlacedPixmap, Position, Regions, pixel_size,
};
use crate::filter::{blur_reduced, upscale_reduced};
use crate::kurbo::Affine;
use alloc::vec::Vec;
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::geometry::PaddingU64;

/// The number of decimations a blur performs on its source at full
/// resolution. A blur that decimates more reads its source averaged over
/// squares of pixels so large that this many decimations remain.
pub(crate) const EXACT_DECIMATIONS: usize = 5;

/// Return `filter` with its parameters in the grid of root target pixels
/// `2^level` wide.
fn filter_at(filter: &FilterData, level: u32) -> FilterData {
    FilterData::new(
        filter.filter.clone(),
        Affine::scale(1.0 / pixel_size(level)) * filter.transform,
    )
}

/// How a filter reads its source and writes its output, in the root grid.
#[derive(Clone, Debug)]
pub(crate) struct Reach {
    filter: FilterData,
    effect: Effect,
    /// By how many steps of two the source the filter reads is coarser than the output.
    reduction: u32,
    /// The decimations of a blur that reads its source at full resolution
    /// line up wherever the pixmaps it runs on start a multiple of
    /// `2^decimations` pixels apart.
    decimations: u32,
    /// How far around each pixel the output of such a blur depends on the
    /// pixmap it runs on, by the taps of its decimations, convolution and
    /// upscaling, which reach beyond the blur's padding.
    cone: u64,
}

#[derive(Clone, Copy, Debug)]
enum Effect {
    /// Filtered as a whole in a pixmap around the source it reads, with
    /// paddings in the grid of that source.
    Whole {
        source_padding: PaddingU64,
        filter_padding: PaddingU64,
    },
    /// Moved by `offset` in the output's grid, see the module documentation,
    /// after a blur that reads `blur_padding` around each pixel in the grid of
    /// the source it reads.
    Displaced {
        offset: Position,
        blur_padding: PaddingU64,
        keeps_source: bool,
    },
}

impl Reach {
    /// Return how `filter` reads and writes.
    pub(crate) fn new(filter: &FilterData) -> Self {
        let (decimations, radius) = match PreparedFilter::new(&filter.filter, &filter.transform) {
            PreparedFilter::GaussianBlur(blur) => (blur.n_decimations, blur.kernel_size / 2),
            PreparedFilter::DropShadow(shadow) => (shadow.n_decimations, shadow.kernel_size / 2),
            PreparedFilter::Flood(_) | PreparedFilter::Offset(_) => (0, 0),
        };
        let reduction = u32::try_from(decimations.saturating_sub(EXACT_DECIMATIONS)).unwrap();
        // Paddings are taken in the grid they apply in, where they stay small
        // however wide the blur is.
        let source = if reduction == 0 {
            filter.clone()
        } else {
            filter_at(filter, reduction)
        };
        let effect = match (filter.displacement(), source.displacement()) {
            (Some(displacement), Some(coarse)) => {
                let (dx, dy) = displacement.offset;
                let offset = |value: f32| Coord::of_f64(value.into());
                Effect::Displaced {
                    offset: (offset(dx), offset(dy)),
                    blur_padding: coarse.blur_padding,
                    keeps_source: displacement.keeps_source,
                }
            }
            _ => Effect::Whole {
                source_padding: source.source_padding,
                filter_padding: source.filter_padding,
            },
        };
        // Only blurs that read their source at full resolution use these,
        // and those decimate at most `EXACT_DECIMATIONS` times.
        let decimations = decimations.min(EXACT_DECIMATIONS);
        Self {
            filter: filter.clone(),
            effect,
            reduction,
            decimations: u32::try_from(decimations).unwrap(),
            cone: (u64::from(radius) + 4) << decimations,
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
                blur_padding,
                keeps_source,
                ..
            } => {
                let kept = if keeps_source { output } else { Bounds::EMPTY };
                (kept, moved.expand(blur_padding))
            }
        }
    }

    /// Return whether the output holds a copy of the source where it is.
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
    fn offset(&self) -> Position {
        match self.effect {
            Effect::Whole { .. } => (Coord::ZERO, Coord::ZERO),
            Effect::Displaced { offset, .. } => offset,
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

/// How a filter layer reads its source and writes its output.
#[derive(Debug)]
pub(crate) struct FilterPlan {
    reach: Reach,
    outputs: Vec<Output>,
    /// The bounds of the outputs, indexed.
    index: Index,
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
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    /// The pixmap the filter runs on, in the grid of the source it reads.
    filtered: Bounds,
    /// The filtered output, in the output's grid before it is moved.
    result: Bounds,
    /// The kept source, in the output's grid before it is moved.
    kept: Bounds,
    /// Where the output has content.
    extent: Bounds,
}

impl Layout {
    /// Return the source to render for the layout at full resolution: the
    /// kept source, and the pixmap the filter runs on unless it reads a
    /// coarser source.
    pub(crate) fn sources(&self, reduced: bool) -> Regions {
        let mut sources = Regions::default();
        sources.insert(self.kept);
        if !reduced {
            sources.insert(self.filtered);
        }
        sources
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.extent.is_empty()
    }
}

/// A pixmap a blur that reads a coarser source has run on, see [`FilterPlan::blur`].
#[derive(Debug)]
pub(crate) struct Blurred {
    pixmap: FilterPixmap,
    bounds: Bounds,
}

impl FilterPlan {
    /// Plan the output of a filter that reads and writes as `reach` inside `outputs`.
    pub(crate) fn new(reach: Reach, outputs: &Regions) -> Self {
        let outputs: Vec<Output> = outputs
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
        Self {
            index: Index::new(outputs.iter().map(|output| output.bounds)),
            reach,
            outputs,
        }
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
                let blurred = (self.reduction() > 0 && !layout.filtered.is_empty())
                    .then(|| self.blur(&layout, filtered, scratch));
                self.compose(&layout, kept, filtered, blurred.as_ref(), scratch, apply)
            })
            .collect()
    }

    /// Return the outputs whose regions overlap `area`, each with the part of
    /// it there laid out, given the bounds of the source as in [`Self::render`].
    ///
    /// The part of an output region is computed exactly as the whole region
    /// is: a filter that reads a coarser source runs on the same pixmap, and
    /// any other runs on the part of that pixmap its output there depends on,
    /// placed where its decimations line up with those of the whole.
    pub(crate) fn parts(
        &self,
        area: Bounds,
        kept: &Index,
        filtered: &Index,
    ) -> Vec<(usize, Layout)> {
        let mut parts: Vec<(usize, Layout)> = self
            .index
            .touching(area)
            .map(|(index, _)| (index, self.part(index, area, kept, filtered)))
            .filter(|(_, layout)| !layout.is_empty())
            .collect();
        parts.sort_unstable_by_key(|&(index, _)| index);
        parts
    }

    fn part(&self, index: usize, area: Bounds, kept: &Index, filtered: &Index) -> Layout {
        let whole = self.layout(&self.outputs[index], kept, filtered);
        let (dx, dy) = self.reach.offset();
        let result = whole.result.intersect(area.translate((-dx, -dy)));
        let kept = whole.kept.intersect(area);
        let pixmap = if result.is_empty() {
            Bounds::EMPTY
        } else if self.reduction() > 0 {
            whole.filtered
        } else {
            let cone = result
                .expand(PaddingU64::new(
                    self.reach.cone,
                    self.reach.cone,
                    self.reach.cone,
                    self.reach.cone,
                ))
                .intersect(whole.filtered);
            let levels = self.reach.decimations;
            let align =
                |start: Coord, value: Coord| start + (value - start).shr(levels).shl(levels);
            Bounds::new(
                align(whole.filtered.x0, cone.x0),
                align(whole.filtered.y0, cone.y0),
                cone.x1,
                cone.y1,
            )
        };
        Layout {
            filtered: pixmap,
            result,
            kept,
            extent: result.translate((dx, dy)).union(kept),
        }
    }

    /// Run the filter, which reads a coarser source, on the pixmap of `layout`
    /// with the source in `filtered`.
    pub(crate) fn blur(
        &self,
        layout: &Layout,
        filtered: &Pieces,
        scratch: &mut ScratchBuffer,
    ) -> Blurred {
        let mut pixmap = filtered.crop(layout.filtered);
        blur_reduced(
            &self.prepared(),
            &mut pixmap.pixmap,
            self.reduction(),
            scratch,
        );
        Blurred {
            pixmap: pixmap.pixmap,
            bounds: pixmap.bounds,
        }
    }

    /// Compute the output of `layout` from the rendered source as in
    /// [`Self::render`]; a filter that reads a coarser source takes it from
    /// `blurred`, which [`Self::blur`] returned for the whole of its output region.
    pub(crate) fn compose(
        &self,
        layout: &Layout,
        kept: &Pieces,
        filtered: &Pieces,
        blurred: Option<&Blurred>,
        scratch: &mut ScratchBuffer,
        apply: &mut dyn FnMut(&mut FilterPixmap, PreparedFilter, &mut ScratchBuffer),
    ) -> PlacedPixmap {
        let mut pixmap = PlacedPixmap::new(layout.extent);
        if !layout.filtered.is_empty() {
            let result = if let Some(blurred) = blurred {
                PlacedPixmap {
                    pixmap: upscale_reduced(
                        &self.prepared(),
                        &blurred.pixmap,
                        self.reduction(),
                        blurred.bounds.origin(),
                        layout.result,
                    ),
                    bounds: layout.result,
                }
            } else {
                let mut pixmap = filtered.crop(layout.filtered);
                apply(&mut pixmap.pixmap, self.prepared(), scratch);
                pixmap.crop(layout.result)
            };
            pixmap.copy_from(&result.translate(self.reach.offset()), layout.extent);
        }
        if !layout.kept.is_empty() {
            pixmap.combine(&kept.crop(layout.kept), layout.kept, source_over);
        }
        pixmap
    }

    fn prepared(&self) -> PreparedFilter {
        let filter = &self.reach.filter;
        PreparedFilter::new(&filter.filter, &filter.transform)
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
            extent: result.translate((dx, dy)).union(kept),
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
