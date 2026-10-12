// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Images averaged over the pixels of a coarse grid.
//!
//! A window replayed in a grid coarser than the root target samples an image
//! once per window pixel, which would skip the image pixels between samples.
//! Instead, the window samples a copy of the image whose samples average the
//! image over a box as wide as a window pixel along each image axis, spaced
//! at most half that width apart, so that sampling the copy bilinearly
//! averages the image pixels each window pixel covers.

use crate::kurbo::{Affine, Point, Vec2};
use crate::peniko::{Extend, ImageQuality, ImageSampler};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::encode::EncodedImage;
use vello_common::paint::ImageSource;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

/// Return a copy of `image`, which paints `pixmap` into a window of `width`
/// × `height` pixels, that samples a version of `pixmap` averaged over each
/// window pixel; or `None` if no window pixel spans more than one image pixel
/// along either image axis.
pub(crate) fn averaged(
    image: &EncodedImage,
    pixmap: &Pixmap,
    width: u16,
    height: u16,
) -> Option<EncodedImage> {
    let [a, b, c, d, _, _] = image.transform.as_coeffs();
    let footprint = (a.abs() + c.abs(), b.abs() + d.abs());
    if !(footprint.0 > 1.0 || footprint.1 > 1.0) || pixmap.width() == 0 || pixmap.height() == 0 {
        return None;
    }
    let corners = [(0, 0), (width, 0), (0, height), (width, height)]
        .map(|(x, y)| image.transform * Point::new(f64::from(x), f64::from(y)));
    // A transform that overflows has no footprint to average over.
    if !corners.iter().all(|corner| corner.is_finite()) {
        return None;
    }
    let range = |coordinate: fn(&Point) -> f64| {
        corners
            .iter()
            .map(coordinate)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
                (low.min(value), high.max(value))
            })
    };
    let x = Axis::new(
        pixmap.width(),
        image.sampler.x_extend,
        footprint.0,
        range(|p| p.x),
    );
    let y = Axis::new(
        pixmap.height(),
        image.sampler.y_extend,
        footprint.1,
        range(|p| p.y),
    );
    let averaged = average(pixmap, &x, &y);
    let transform = Affine::new([
        1.0 / x.stride,
        0.0,
        0.0,
        1.0 / y.stride,
        -(x.origin + x.start) / x.stride,
        -(y.origin + y.start) / y.stride,
    ]) * image.transform;
    let [a, b, c, d, _, _] = transform.as_coeffs();
    Some(EncodedImage {
        source: ImageSource::Pixmap(Arc::new(averaged)),
        sampler: ImageSampler {
            x_extend: x.extend(),
            y_extend: y.extend(),
            quality: ImageQuality::Medium,
            ..image.sampler
        },
        may_have_transparency: image.may_have_transparency,
        transform,
        x_advance: Vec2::new(a, b),
        y_advance: Vec2::new(c, d),
        tint: image.tint,
    })
}

/// Where the samples of the averaged image lie along one image axis, in
/// image pixels.
#[derive(Debug)]
struct Axis {
    /// The image's size along the axis.
    size: u16,
    extend: Extend,
    /// The width of the box each sample averages.
    width: f64,
    /// A whole number of periods of the extended image, which the positions
    /// below are relative to.
    origin: f64,
    /// The position of the first sample's left edge.
    start: f64,
    /// The distance between samples.
    stride: f64,
    count: u16,
    /// Whether the samples cover one period of the extended image, which the
    /// averaged image then repeats like the image; otherwise they cover the
    /// positions the window samples and their surroundings.
    periodic: bool,
    /// The pixels of the extended image the samples average.
    pixels: Range<i64>,
}

impl Axis {
    /// `footprint` is the extent of a window pixel along the axis, and
    /// `positions` the range of positions the window samples.
    fn new(size: u16, extend: Extend, footprint: f64, positions: (f64, f64)) -> Self {
        let length = f64::from(size);
        let width = footprint.max(1.0);
        let stride = (width / 2.0).max(1.0);
        let period = match extend {
            Extend::Pad => None,
            Extend::Repeat => Some(length),
            Extend::Reflect => Some(2.0 * length),
        };
        let (low, high) = positions;
        if let Some(period) = period {
            let origin = (low / period).floor() * period;
            if high - low + width + 4.0 * stride >= period {
                // Whole samples per image, so that the averaged image repeats with the image.
                let count = (length / stride).ceil();
                let stride = length / count;
                return Self {
                    size,
                    extend,
                    width: width.max(stride),
                    origin,
                    start: 0.0,
                    stride,
                    count: count as u16,
                    periodic: true,
                    pixels: 0..period as i64,
                };
            }
            // Positions too far out for `f64` to place within a period are
            // sampled at its start.
            let low = (low - origin).clamp(0.0, period);
            return Self::covering(
                size,
                extend,
                width,
                stride,
                origin,
                (low, low + high - positions.0),
            );
        }
        // Past half a box beyond the image, the samples repeat the edge pixels.
        let margin = width / 2.0 + 2.0 * stride;
        let clamp = |position: f64| position.clamp(-margin, length + margin);
        Self::covering(size, extend, width, stride, 0.0, (clamp(low), clamp(high)))
    }

    /// Return samples covering `positions`, relative to `origin`, with a
    /// sample to spare on each side.
    fn covering(
        size: u16,
        extend: Extend,
        width: f64,
        stride: f64,
        origin: f64,
        positions: (f64, f64),
    ) -> Self {
        let (low, high) = positions;
        let stride = stride.max((high - low) / f64::from(u16::MAX - 8));
        let first = (low / stride).floor() - 2.0;
        let count = (high / stride).ceil() - first + 1.0;
        // A box narrower than the stride would skip pixels between samples.
        let width = width.max(stride);
        let start = first * stride;
        let pixels = (start - width / 2.0).floor() as i64
            ..(start + count * stride + width / 2.0).ceil() as i64;
        let pixels = match extend {
            Extend::Pad => {
                let clamp = |pixel: i64| pixel.clamp(0, i64::from(size) - 1);
                clamp(pixels.start)..clamp(pixels.end - 1) + 1
            }
            Extend::Repeat | Extend::Reflect => pixels,
        };
        Self {
            size,
            extend,
            width,
            origin,
            start,
            stride,
            count: count as u16,
            periodic: false,
            pixels,
        }
    }

    /// Return how the averaged image extends past its samples.
    fn extend(&self) -> Extend {
        if self.periodic {
            self.extend
        } else {
            Extend::Pad
        }
    }

    /// Return the image pixel at `pixel` of the extended image.
    fn pixel(&self, pixel: i64) -> u16 {
        let size = i64::from(self.size);
        let pixel = match self.extend {
            Extend::Pad => pixel.clamp(0, size - 1),
            Extend::Repeat => pixel.rem_euclid(size),
            Extend::Reflect => {
                let pixel = pixel.rem_euclid(2 * size);
                if pixel < size {
                    pixel
                } else {
                    2 * size - 1 - pixel
                }
            }
        };
        pixel as u16
    }

    /// Return the average of sample `index` over the line of `sums`.
    fn sample(&self, sums: &Sums, index: u16) -> [f64; 4] {
        let center = self.start + (f64::from(index) + 0.5) * self.stride;
        let (low, high) = (center - self.width / 2.0, center + self.width / 2.0);
        let sum = if self.periodic {
            sums.periodic(low, high)
        } else {
            sub(sums.to(high), sums.to(low))
        };
        sum.map(|value| value / self.width)
    }
}

/// Running sums of a line of pixels of the extended image.
struct Sums {
    /// The position of the first pixel.
    start: i64,
    /// The sums of the first `n` pixels for each `n` up to all of them.
    sums: Vec<[f64; 4]>,
}

impl Sums {
    fn new(start: i64) -> Self {
        Self {
            start,
            sums: Vec::new(),
        }
    }

    fn fill(&mut self, pixels: impl Iterator<Item = [f64; 4]>) {
        self.sums.clear();
        self.sums.push([0.0; 4]);
        let mut sum = [0.0; 4];
        for pixel in pixels {
            sum = add(sum, pixel);
            self.sums.push(sum);
        }
    }

    /// Return the sum of the line from its first pixel to `position`,
    /// extending the first and last pixels past the ends of the line.
    fn to(&self, position: f64) -> [f64; 4] {
        let offset = position - self.start as f64;
        let last = self.sums.len() - 2;
        let index = offset.floor().clamp(0.0, last as f64) as usize;
        let pixel = sub(self.sums[index + 1], self.sums[index]);
        add(
            self.sums[index],
            pixel.map(|value| value * (offset - index as f64)),
        )
    }

    /// Return the sum from `low` to `high` of the line repeated with the
    /// period of its length, which starts at zero.
    fn periodic(&self, low: f64, high: f64) -> [f64; 4] {
        let period = (self.sums.len() - 1) as f64;
        let total = self.sums[self.sums.len() - 1];
        let to = |position: f64| {
            let periods = (position / period).floor();
            add(
                total.map(|value| value * periods),
                self.to(position - periods * period),
            )
        };
        let shift = (low / period).floor() * period;
        sub(to(high - shift), to(low - shift))
    }
}

/// Return `pixmap` averaged at the samples of `x` and `y`.
fn average(pixmap: &Pixmap, x: &Axis, y: &Axis) -> Pixmap {
    let pixel = |x: u16, y: u16| {
        let pixel = pixmap.sample(x, y);
        [pixel.r, pixel.g, pixel.b, pixel.a].map(f64::from)
    };
    let columns = usize::from(x.count);
    let mut rows = Vec::new();
    let mut sums = Sums::new(x.pixels.start);
    for row in y.pixels.clone() {
        let row = y.pixel(row);
        sums.fill(x.pixels.clone().map(|column| pixel(x.pixel(column), row)));
        rows.extend((0..x.count).map(|index| x.sample(&sums, index).map(|value| value as f32)));
    }
    let mut averaged = Pixmap::new(x.count, y.count);
    let mut sums = Sums::new(y.pixels.start);
    for column in 0..columns {
        sums.fill(
            rows[column..]
                .iter()
                .step_by(columns)
                .map(|pixel| pixel.map(f64::from)),
        );
        for index in 0..y.count {
            let [r, g, b, a] = y
                .sample(&sums, index)
                .map(|value| (value + 0.5).clamp(0.0, 255.0) as u8);
            averaged.data_mut()[usize::from(index) * columns + column] = PremulRgba8 {
                r: r.min(a),
                g: g.min(a),
                b: b.min(a),
                a,
            };
        }
    }
    averaged.set_may_have_transparency(pixmap.may_have_transparency());
    averaged
}

fn add(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn sub(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}
