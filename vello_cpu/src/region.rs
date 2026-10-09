// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::pixmap::PixmapMut;
use vello_common::tile::Tile;

/// Mutable byte view shared by external targets and private wide filter buffers.
#[derive(Debug)]
pub(crate) struct RasterTarget<'a> {
    width: u32,
    height: u32,
    data: &'a mut [u8],
}

impl<'a> RasterTarget<'a> {
    pub(crate) fn new(width: u32, height: u32, data: &'a mut [u8]) -> Self {
        let bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(COLOR_COMPONENTS))
            .expect("raster target area exceeds address space");
        assert_eq!(
            bytes,
            data.len(),
            "raster target byte length must match dimensions"
        );
        Self {
            width,
            height,
            data,
        }
    }
    pub(crate) fn from_pixmap(pixmap: &'a mut PixmapMut<'_>) -> Self {
        Self::new(
            u32::from(pixmap.width()),
            u32::from(pixmap.height()),
            pixmap.data_mut(),
        )
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

/// A view into a part of a single strip row of a pixmap.
#[derive(Default, Debug)]
pub struct Region<'a> {
    pub(crate) row_idx: usize,
    width: u32,
    pub(crate) height: u32,
    areas: [&'a mut [u8]; Tile::HEIGHT_U32 as usize],
}

impl<'a> Region<'a> {
    #[doc(hidden)]
    pub fn new(pixmap: &'a mut PixmapMut<'_>, rect: RectU16) -> Self {
        Self::new_from_row(pixmap, rect, 0)
    }

    pub(crate) fn new_from_row(
        pixmap: &'a mut PixmapMut<'_>,
        rect: RectU16,
        row_idx: usize,
    ) -> Self {
        let width = u32::from(rect.width());
        let height = u32::from(rect.height()).min(Tile::HEIGHT_U32);
        let row_stride = (pixmap.width() as usize) * COLOR_COMPONENTS;
        let start_offset = (rect.y0 as usize) * row_stride;
        let x_offset = (rect.x0 as usize) * COLOR_COMPONENTS;
        let buffer = pixmap.data_mut();
        Self::from_rows(
            row_idx,
            width,
            height,
            row_stride,
            x_offset,
            &mut buffer[start_offset..],
        )
    }

    pub(crate) fn row_mut(&mut self, y: u32) -> &mut [u8] {
        self.areas[y as usize]
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    /// Return a horizontal sub-span of the region.
    pub(crate) fn sub_span(&mut self, x: u32, width: u32) -> Region<'_> {
        let x_offset = (x as usize) * COLOR_COMPONENTS;
        let row_width_bytes = (width as usize) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT_U32 as usize] =
            [&mut [], &mut [], &mut [], &mut []];

        for (source, area) in self
            .areas
            .iter_mut()
            .take(self.height as usize)
            .zip(areas.iter_mut())
        {
            let (_, source) = source.split_at_mut(x_offset);
            let (source, _) = source.split_at_mut(row_width_bytes);
            *area = source;
        }

        Region {
            row_idx: self.row_idx,
            width,
            height: self.height,
            areas,
        }
    }

    pub(crate) fn areas(&mut self) -> &mut [&'a mut [u8]; Tile::HEIGHT_U32 as usize] {
        &mut self.areas
    }

    fn from_rows(
        row_idx: usize,
        width: u32,
        height: u32,
        row_stride: usize,
        x_offset: usize,
        mut rows: &'a mut [u8],
    ) -> Self {
        let row_width_bytes = (width as usize) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT_U32 as usize] =
            [&mut [], &mut [], &mut [], &mut []];

        for area in areas.iter_mut().take(height as usize) {
            let (row, rest) = rows.split_at_mut(row_stride);
            let (_, row) = row.split_at_mut(x_offset);
            let (row, _) = row.split_at_mut(row_width_bytes);
            *area = row;
            rows = rest;
        }

        Self {
            row_idx,
            width,
            height,
            areas,
        }
    }
}

/// Split a pixmap into an array of regions.
pub(crate) struct Regions<'a> {
    regions: Vec<Region<'a>>,
}

impl<'a> Regions<'a> {
    pub(crate) fn new(
        target: &'a mut RasterTarget<'_>,
        scene_size: (u32, u32),
        offset: (u32, u32),
        row_count: usize,
    ) -> Self {
        let (dst_x, dst_y) = offset;

        let (scene_width, scene_height) = scene_size;
        let width = scene_width.min(target.width().saturating_sub(dst_x));
        let height = scene_height.min(target.height().saturating_sub(dst_y));

        if width == 0 || height == 0 {
            return Self {
                regions: Vec::new(),
            };
        }

        let row_count = row_count.min((height as usize).div_ceil(Tile::HEIGHT_U32 as usize));
        let stride = (target.width() as usize) * COLOR_COMPONENTS;
        let x_offset = (dst_x as usize) * COLOR_COMPONENTS;
        let render_bytes = (height as usize) * stride;
        let target = target.data_mut();
        let mut remaining = &mut target[(dst_y as usize) * stride..][..render_bytes];
        let mut regions = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            let row_y = row_idx as u32 * Tile::HEIGHT_U32;
            let row_height = (height - row_y).min(Tile::HEIGHT_U32);
            let band_len = (row_height as usize) * stride;
            let (buffer, rest) = remaining.split_at_mut(band_len);
            regions.push(Region::from_rows(
                row_idx, width, row_height, stride, x_offset, buffer,
            ));
            remaining = rest;
        }

        Self { regions }
    }

    pub(crate) fn update(&mut self, func: impl FnMut(&mut Region<'_>)) {
        self.regions.iter_mut().for_each(func);
    }

    #[cfg(feature = "multithreading")]
    pub(crate) fn update_par(&mut self, func: impl Fn(&mut Region<'_>) + Send + Sync) {
        use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

        self.regions.par_iter_mut().for_each(func);
    }
}

#[cfg(test)]
mod tests {
    use super::Regions;
    use vello_common::pixmap::Pixmap;

    #[test]
    fn regions_with_off_target_offsets_do_not_panic() {
        for offset in [(20, 0), (0, 20)] {
            let mut pixmap = Pixmap::new(10, 10);
            let mut pixmap = pixmap.as_mut();
            let mut target = super::RasterTarget::from_pixmap(&mut pixmap);
            let _regions = Regions::new(&mut target, (4, 4), offset, 1);
        }
    }
}
