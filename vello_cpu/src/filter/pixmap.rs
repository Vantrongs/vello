// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Private storage for filter sources, whose halo may exceed a physical target.

use alloc::{vec, vec::Vec};
use vello_common::peniko::ImageAlphaType;
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::{PixelMetadata, Pixmap, PixmapMut};

#[derive(Debug, Clone)]
pub(crate) struct FilterPixmap {
    width: u32,
    height: u32,
    pixels: Vec<PremulRgba8>,
}

impl FilterPixmap {
    fn pixel_count(width: u32, height: u32) -> usize {
        let len = (width as usize)
            .checked_mul(height as usize)
            .expect("filter buffer area exceeds address space");
        len.checked_mul(size_of::<PremulRgba8>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .expect("filter buffer allocation exceeds address space");
        len
    }

    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![PremulRgba8::from_u32(0); Self::pixel_count(width, height)],
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let len = Self::pixel_count(width, height);
        self.pixels.resize(len, PremulRgba8::from_u32(0));
        self.width = width;
        self.height = height;
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }
    pub(crate) fn height(&self) -> u32 {
        self.height
    }
    pub(crate) fn data(&self) -> &[PremulRgba8] {
        &self.pixels
    }
    pub(crate) fn data_mut(&mut self) -> &mut [PremulRgba8] {
        &mut self.pixels
    }
    pub(crate) fn sample(&self, x: u32, y: u32) -> PremulRgba8 {
        self.data()[y as usize * self.width as usize + x as usize]
    }
    pub(crate) fn set_pixel(&mut self, x: u32, y: u32, pixel: PremulRgba8) {
        self.pixels[y as usize * self.width as usize + x as usize] = pixel;
    }
    /// Return a render target over the pixmap, which must fit the `u16` domain.
    pub(crate) fn as_target(&mut self) -> PixmapMut<'_> {
        let narrow = |size: u32| u16::try_from(size).expect("render targets fit the u16 domain");
        PixmapMut::new(
            narrow(self.width),
            narrow(self.height),
            bytemuck::cast_slice_mut(&mut self.pixels),
        )
        .expect("the buffer holds width * height pixels")
    }

    /// Convert into a pixmap without copying, which must fit the `u16` domain.
    pub(crate) fn into_pixmap(self) -> Pixmap {
        let narrow = |size: u32| u16::try_from(size).expect("pixmaps fit the u16 domain");
        Pixmap::from_parts(
            bytemuck::cast_vec(self.pixels),
            narrow(self.width),
            narrow(self.height),
            PixelMetadata {
                alpha_type: ImageAlphaType::AlphaPremultiplied,
                may_have_transparency: true,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::FilterPixmap;

    #[test]
    #[should_panic(expected = "filter buffer")]
    fn rejects_unaddressable_area_before_allocation() {
        FilterPixmap::pixel_count(u32::MAX, u32::MAX);
    }
}
