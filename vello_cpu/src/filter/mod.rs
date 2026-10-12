// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects implementation for `vello_cpu`.
//!
//! This module provides CPU-based implementations of SVG filter effects,
//! supporting both low-precision (u8) and high-precision (f32) rendering paths.
//! Filters are applied to rendered layer pixmaps and may use scratch storage for
//! intermediate buffers.

pub(crate) mod averaged_image;
pub(crate) mod context;
mod drop_shadow;
mod flood;
mod gaussian_blur;
pub(crate) mod pixmap;
pub(crate) mod placed;
pub(crate) mod plan;

use crate::filter::pixmap::FilterPixmap;
use crate::filter::placed::Bounds;
use context::ScratchBuffer;
use vello_common::filter::PreparedFilter;

/// Trait for filter effects that can be applied to layers.
///
/// Each filter implements this trait with both u8 and f32 variants
/// to support different rendering backends and precision requirements.
/// The low-precision path (`execute_lowp`) uses 8-bit color channels for
/// better performance, while the high-precision path (`execute_highp`)
/// uses 32-bit floating-point for higher quality at the cost of speed.
pub(crate) trait FilterEffect {
    /// Apply the low-precision (u8) version of the filter.
    ///
    /// # Arguments
    /// * `pixmap` - The target pixmap containing rendering metadata
    /// * `filter_scratch` - Reusable scratch storage for intermediate buffers
    fn execute_lowp(&self, pixmap: &mut FilterPixmap, filter_scratch: &mut ScratchBuffer);

    /// Apply the high-precision (f32) version of the filter.
    ///
    /// # Arguments
    /// * `pixmap` - The target pixmap containing rendering metadata
    /// * `filter_scratch` - Reusable scratch storage for intermediate buffers
    fn execute_highp(&self, pixmap: &mut FilterPixmap, filter_scratch: &mut ScratchBuffer);
}

/// Apply the low-precision (u8) version of a filter effect to a layer.
///
/// This function dispatches filter primitives from a filter graph to their
/// corresponding CPU implementations using 8-bit color channels.
///
/// # Arguments
/// * `filter` - The filter prepared for the layer's transform
/// * `pixmap` - The target pixmap containing rendering metadata
/// * `filter_scratch` - Reusable scratch storage for intermediate buffers
///
/// # Limitations
/// Currently only supports filter graphs with a single primitive.
/// Multi-primitive filter graphs are not yet implemented.
pub(crate) fn filter_lowp(
    filter: PreparedFilter,
    pixmap: &mut FilterPixmap,
    filter_scratch: &mut ScratchBuffer,
) {
    match filter {
        PreparedFilter::Flood(flood) => {
            flood.execute_lowp(pixmap, filter_scratch);
        }
        PreparedFilter::GaussianBlur(blur) => {
            blur.execute_lowp(pixmap, filter_scratch);
        }
        // An offset only moves pixels, which happens when the layer's output
        // is assembled, see `plan`.
        PreparedFilter::Offset(_) => {}
        PreparedFilter::DropShadow(drop_shadow) => {
            drop_shadow.execute_lowp(pixmap, filter_scratch);
        }
    }
}

/// Apply the high-precision (f32) version of a filter effect to a layer.
///
/// This function dispatches filter primitives from a filter graph to their
/// corresponding CPU implementations using 32-bit floating-point color channels.
///
/// # Arguments
/// * `filter` - The filter prepared for the layer's transform
/// * `pixmap` - The target pixmap containing rendering metadata
/// * `filter_scratch` - Reusable scratch storage for intermediate buffers
///
/// # Limitations
/// Currently only supports filter graphs with a single primitive.
/// Multi-primitive filter graphs are not yet implemented.
pub(crate) fn filter_highp(
    filter: PreparedFilter,
    pixmap: &mut FilterPixmap,
    filter_scratch: &mut ScratchBuffer,
) {
    match filter {
        PreparedFilter::Flood(flood) => {
            flood.execute_highp(pixmap, filter_scratch);
        }
        PreparedFilter::GaussianBlur(blur) => {
            blur.execute_highp(pixmap, filter_scratch);
        }
        // An offset only moves pixels, which happens when the layer's output
        // is assembled, see `plan`.
        PreparedFilter::Offset(_) => {}
        PreparedFilter::DropShadow(drop_shadow) => {
            drop_shadow.execute_highp(pixmap, filter_scratch);
        }
    }
}

/// Apply a blur or drop shadow to `pixmap`, whose pixels each cover
/// `2^levels` pixels of the output on both axes and which starts at `origin`
/// in its grid, and return the output inside `crop`, in the grid of the output.
///
/// Both kernels apply blurs and drop shadows at low precision, so this serves both.
pub(crate) fn filter_reduced(
    filter: PreparedFilter,
    pixmap: FilterPixmap,
    levels: u32,
    origin: (i64, i64),
    crop: Bounds,
    filter_scratch: &mut ScratchBuffer,
) -> FilterPixmap {
    let scratch = filter_scratch.get_scratch_buffer(pixmap.width(), pixmap.height());
    match filter {
        PreparedFilter::GaussianBlur(blur) => gaussian_blur::apply_reduced_blur(
            pixmap,
            scratch,
            blur.n_decimations,
            &blur.kernel[..usize::from(blur.kernel_size)],
            blur.edge_mode,
            levels,
            origin,
            crop,
        ),
        PreparedFilter::DropShadow(shadow) => {
            let mut output = gaussian_blur::apply_reduced_blur(
                pixmap,
                scratch,
                shadow.n_decimations,
                &shadow.kernel[..usize::from(shadow.kernel_size)],
                shadow.edge_mode,
                levels,
                origin,
                crop,
            );
            drop_shadow::colorize(&mut output, shadow.color);
            output
        }
        PreparedFilter::Flood(_) | PreparedFilter::Offset(_) => {
            unreachable!("only blurs read a reduced source")
        }
    }
}
