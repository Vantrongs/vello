// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::filter::pixmap::FilterPixmap;
use crate::fine::macros::u8x16_painter;
use alloc::sync::Arc;
use vello_common::fearless_simd::{Simd, SimdBase, u8x16};
use vello_common::geometry::RectU32;
use vello_common::peniko::color::PremulRgba8;

/// Filter composition only translates by whole pixels and uses nearest-neighbor Pad sampling.
#[derive(Debug)]
#[doc(hidden)]
pub struct FilterPaint {
    pub(crate) pixmap: Arc<FilterPixmap>,
    pub(crate) src_offset: (i64, i64),
    pub(crate) dest_bbox: RectU32,
}

pub(crate) struct FilterPainter<'a, S: Simd> {
    paint: &'a FilterPaint,
    x: i64,
    rows: [Option<u32>; 4],
    dest_x: usize,
    simd: S,
}

impl<'a, S: Simd> FilterPainter<'a, S> {
    pub(crate) fn new(simd: S, paint: &'a FilterPaint, x: usize, y: u32) -> Self {
        let max_y = i64::from(paint.pixmap.height()) - 1;
        let rows = core::array::from_fn(|row| {
            let dest_y = u64::from(y) + row as u64;
            (dest_y >= u64::from(paint.dest_bbox.y0) && dest_y < u64::from(paint.dest_bbox.y1))
                .then(|| (dest_y as i64 + paint.src_offset.1).clamp(0, max_y) as u32)
        });
        Self {
            paint,
            x: x as i64 + paint.src_offset.0,
            dest_x: x,
            rows,
            simd,
        }
    }
}

impl<S: Simd> Iterator for FilterPainter<'_, S> {
    type Item = u8x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let pixmap = &self.paint.pixmap;
        let x = self.x.clamp(0, i64::from(pixmap.width()) - 1) as u32;
        let in_bounds = self.dest_x >= self.paint.dest_bbox.x0 as usize
            && self.dest_x < self.paint.dest_bbox.x1 as usize;
        let pixels = self.rows.map(|y| match y {
            Some(y) if in_bounds => pixmap.sample(x, y),
            _ => PremulRgba8::from_u32(0),
        });
        self.x += 1;
        self.dest_x += 1;
        Some(u8x16::from_slice(self.simd, bytemuck::cast_slice(&pixels)))
    }
}

u8x16_painter!(FilterPainter<'_, S>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fine::Painter;
    use crate::fine::common::image::PlainNNImagePainter;
    use vello_common::encode::EncodedImage;
    use vello_common::fearless_simd::Fallback;
    use vello_common::kurbo::{Affine, Vec2};
    use vello_common::paint::ImageSource;
    use vello_common::peniko::color::PremulRgba8;
    use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
    use vello_common::pixmap::Pixmap;

    #[test]
    fn integer_filter_sampling_matches_image_sampling_and_pad_edges() {
        let simd = Fallback::new();
        let mut ordinary = Pixmap::new(7, 5);
        for (idx, pixel) in ordinary.data_mut().iter_mut().enumerate() {
            *pixel = PremulRgba8 {
                r: idx as u8,
                g: (idx * 2) as u8,
                b: (idx * 3) as u8,
                a: 127,
            };
        }
        let mut wide = FilterPixmap::new(7, 5);
        wide.data_mut().copy_from_slice(ordinary.data());
        let ordinary = Arc::new(ordinary);
        let wide = Arc::new(wide);
        for x in [0, 3, 65_530, 70_000] {
            for y in [0, 3, 65_535, 70_000] {
                for src_offset in [(-8, -4), (0, 0), (5, 3), (-65_532, -65_536)] {
                    let paint = FilterPaint {
                        pixmap: wide.clone(),
                        src_offset,
                        dest_bbox: RectU32::new(0, 0, u32::MAX, u32::MAX),
                    };
                    let image = EncodedImage {
                        source: ImageSource::Pixmap(ordinary.clone()),
                        sampler: ImageSampler {
                            x_extend: Extend::Pad,
                            y_extend: Extend::Pad,
                            quality: ImageQuality::Low,
                            alpha: 1.0,
                        },
                        may_have_transparency: true,
                        transform: Affine::translate((src_offset.0 as f64, src_offset.1 as f64)),
                        x_advance: Vec2::new(1.0, 0.0),
                        y_advance: Vec2::new(0.0, 1.0),
                        tint: None,
                    };
                    let mut expected_u8 = [0; 16 * 16];
                    let mut actual_u8 = [0; 16 * 16];
                    PlainNNImagePainter::new(
                        simd,
                        &image,
                        &ordinary,
                        x as f64 + 0.5,
                        f64::from(y) + 0.5,
                    )
                    .paint_u8(&mut expected_u8);
                    FilterPainter::new(simd, &paint, x, y).paint_u8(&mut actual_u8);
                    assert_eq!(
                        actual_u8, expected_u8,
                        "x={x}, y={y}, offset={src_offset:?}"
                    );
                    let mut expected_f32 = [0.0; 16 * 16];
                    let mut actual_f32 = [0.0; 16 * 16];
                    PlainNNImagePainter::new(
                        simd,
                        &image,
                        &ordinary,
                        x as f64 + 0.5,
                        f64::from(y) + 0.5,
                    )
                    .paint_f32(&mut expected_f32);
                    FilterPainter::new(simd, &paint, x, y).paint_f32(&mut actual_f32);
                    assert_eq!(
                        actual_f32, expected_f32,
                        "x={x}, y={y}, offset={src_offset:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn filter_sampling_reaches_pixels_beyond_u16_on_both_axes() {
        let simd = Fallback::new();
        for (width, height, x, y) in [(70_003, 5, 69_997, 1), (7, 70_003, 3, 69_997)] {
            let mut pixmap = FilterPixmap::new(width, height);
            for (idx, pixel) in pixmap.data_mut().iter_mut().enumerate() {
                *pixel = PremulRgba8 {
                    r: (idx % 251) as u8,
                    g: (idx % 199) as u8,
                    b: (idx % 113) as u8,
                    a: 255,
                };
            }
            let paint = FilterPaint {
                pixmap: Arc::new(pixmap),
                src_offset: (0, 0),
                dest_bbox: RectU32::new(0, 0, u32::MAX, u32::MAX),
            };
            let mut bytes = [0; 16 * 16];
            FilterPainter::new(simd, &paint, x as usize, y).paint_u8(&mut bytes);
            for (column, chunk) in bytes.chunks_exact(16).enumerate() {
                let expected = core::array::from_fn::<_, 4, _>(|row| {
                    paint.pixmap.sample(
                        (x + column as u32).min(width - 1),
                        (y + row as u32).min(height - 1),
                    )
                });
                assert_eq!(chunk, bytemuck::cast_slice(&expected));
            }
        }
    }

    #[test]
    fn integer_offset_does_not_pad_outside_unaligned_destination() {
        let simd = Fallback::new();
        let mut pixmap = FilterPixmap::new(4, 4);
        let color = PremulRgba8 {
            r: 25,
            g: 30,
            b: 40,
            a: 90,
        };
        pixmap.data_mut().fill(color);
        let paint = FilterPaint {
            pixmap: Arc::new(pixmap),
            src_offset: (-1, -3),
            dest_bbox: RectU32::new(1, 3, 5, 7),
        };
        for y in [0, 4] {
            let mut bytes = [0; 8 * 16];
            FilterPainter::new(simd, &paint, 0, y).paint_u8(&mut bytes);
            for (x, chunk) in bytes.chunks_exact(16).enumerate() {
                for (row, rgba) in chunk.chunks_exact(4).enumerate() {
                    let y = y as usize + row;
                    let expected = if (1..5).contains(&x) && (3..7).contains(&y) {
                        color
                    } else {
                        PremulRgba8::from_u32(0)
                    };
                    assert_eq!(rgba, bytemuck::bytes_of(&expected), "x={x}, y={y}");
                }
            }
        }
    }
}
