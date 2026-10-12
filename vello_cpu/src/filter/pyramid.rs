// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Parts of the blur that [`apply_blur`] computes on a whole pixmap, computed
//! from only the source they depend on, for pixmaps of any size.
//!
//! [`apply_blur`] decimates its source, convolves the coarsest level and
//! upscales the result back, each step a horizontal and then a vertical pass
//! that rounds to whole channel values and reads past the edges of its level
//! as the edge mode says. Each pixel of a pass depends on a few pixels of the
//! level it reads, so the pixels of an area are computed from the positions
//! each pass reads, gathered along each axis, down to the source. Decimated
//! levels are computed in chunks and only where they can have content, so
//! memory stays bounded however large the pixmap and the blur are.
//!
//! [`apply_blur`]: super::gaussian_blur::apply_blur

use super::gaussian_blur::{decimate_weighted, interpolate_25_75, interpolate_75_25};
use crate::filter::pixmap::FilterPixmap;
use crate::filter::placed::{Bounds, Coord, Regions};
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::filter::gaussian_blur::MAX_KERNEL_SIZE;
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;

/// The largest side of a chunk of a decimated level computed at once.
const CHUNK: usize = 256;

/// The number of pixels of the coarsest level kept for later areas.
const CACHED_PIXELS: usize = 1 << 22;

const TRANSPARENT: PremulRgba8 = PremulRgba8 {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
};

/// A blur as [`apply_blur`](super::gaussian_blur::apply_blur) applies it.
#[derive(Clone, Debug)]
pub(crate) struct Blur {
    pub(crate) decimations: u32,
    pub(crate) kernel: Vec<f32>,
    pub(crate) edge_mode: EdgeMode,
}

impl Blur {
    /// Return how far the source that a pixel of the output depends on
    /// reaches from it, besides what the edge mode brings in from elsewhere.
    ///
    /// A pixel reads two pixels of the next coarser level on each upscaling,
    /// so at most two pixels either side at the coarsest level, which the
    /// convolution widens by its radius, and each decimated pixel `i` reads
    /// the pixels `2i - 1..=2i + 2` of the level before it.
    pub(crate) fn support(&self) -> Coord {
        Coord::from(self.kernel.len() / 2 + 5).shl(self.decimations)
    }
}

/// Positions along one axis of a level, in runs of consecutive positions.
#[derive(Clone, Debug, Default, PartialEq)]
struct Line {
    runs: Vec<(Coord, usize)>,
}

impl Line {
    fn single(start: Coord, len: usize) -> Self {
        Self {
            runs: vec![(start, len)],
        }
    }

    /// Return the line through the positions of `runs`, merged in order.
    fn merged(mut runs: Vec<(Coord, usize)>) -> Self {
        runs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut merged: Vec<(Coord, usize)> = Vec::with_capacity(runs.len());
        for (start, len) in runs {
            if let Some((last, last_len)) = merged.last_mut() {
                let end = &*last + i64::try_from(*last_len).unwrap();
                if start <= end {
                    let new_end = &start + i64::try_from(len).unwrap();
                    if new_end > end {
                        *last_len = (new_end - &*last).to_usize();
                    }
                    continue;
                }
            }
            merged.push((start, len));
        }
        Self { runs: merged }
    }

    fn len(&self) -> usize {
        self.runs.iter().map(|(_, len)| len).sum()
    }

    /// Return the index of `position` among the positions of the line.
    fn index_of(&self, position: &Coord) -> Option<usize> {
        let mut index = 0;
        for (start, len) in &self.runs {
            if position >= start {
                let offset = position - start;
                if offset < Coord::from(*len) {
                    return Some(index + offset.to_usize());
                }
            }
            index += len;
        }
        None
    }

    /// Return the runs split into pieces of at most `chunk` positions, each
    /// with the index of its first position.
    fn chunks(&self, chunk: usize) -> Vec<(usize, Coord, usize)> {
        let mut chunks = Vec::new();
        let mut index = 0;
        for (start, len) in &self.runs {
            let pieces = len.div_ceil(chunk);
            for piece in 0..pieces {
                let from = len * piece / pieces;
                let to = len * (piece + 1) / pieces;
                chunks.push((
                    index + from,
                    start + i64::try_from(from).unwrap(),
                    to - from,
                ));
            }
            index += len;
        }
        chunks
    }
}

/// Return where a pass on a level of `size` reads `position`, or `None` where
/// it reads transparent, as `extend` in the blur does.
fn extend(position: &Coord, size: &Coord, edge_mode: EdgeMode) -> Option<Coord> {
    if !position.is_negative() && position < size {
        return Some(position.clone());
    }
    match edge_mode {
        EdgeMode::None => None,
        EdgeMode::Duplicate => Some(if position.is_negative() {
            Coord::ZERO
        } else {
            size - 1
        }),
        EdgeMode::Wrap => Some(position.rem_euclid(size)),
        EdgeMode::Mirror => {
            let period = size.shl(1);
            let folded = position.rem_euclid(&period);
            Some(if &folded >= size {
                period - folded - 1
            } else {
                folded
            })
        }
    }
}

/// A pass of the blur along one axis.
#[derive(Clone, Copy, Debug)]
enum Stage<'a> {
    /// Each pixel `i` is `[1, 3, 3, 1] / 8` of the pixels `2i - 1..=2i + 2`.
    Decimate,
    Convolve(&'a [f32]),
    /// Each pixel interpolates the pixel of the coarser level it lies in and
    /// that pixel's neighbor on the side of its half.
    Upscale,
}

impl Stage<'_> {
    fn taps(self) -> usize {
        match self {
            Self::Decimate => 4,
            Self::Convolve(kernel) => kernel.len(),
            Self::Upscale => 2,
        }
    }

    /// Return the first position that the pixel at `position` reads.
    fn first(self, position: &Coord) -> Coord {
        match self {
            Self::Decimate => position.shl(1) - 1,
            Self::Convolve(kernel) => position - i64::try_from(kernel.len() / 2).unwrap(),
            Self::Upscale => position.shr(1) - 1 + i64::from(position.is_odd()),
        }
    }

    /// Return how much later than the pixel at the start of a run, whose
    /// parity is `odd`, the pixel `index` places on reads first.
    fn step(self, odd: bool, index: usize) -> usize {
        match self {
            Self::Decimate => 2 * index,
            Self::Convolve(_) => index,
            Self::Upscale => {
                let parity = usize::from(odd);
                let shifted = parity + index;
                (shifted >> 1) + (shifted & 1) - parity
            }
        }
    }

    fn combine(self, pixels: &[PremulRgba8], odd: bool) -> PremulRgba8 {
        match self {
            Self::Decimate => decimate_weighted(pixels[0], pixels[1], pixels[2], pixels[3]),
            Self::Convolve(kernel) => {
                let mut rgba = [0.0_f32; 4];
                for (pixel, &weight) in pixels.iter().zip(kernel) {
                    rgba[0] += pixel.r as f32 * weight;
                    rgba[1] += pixel.g as f32 * weight;
                    rgba[2] += pixel.b as f32 * weight;
                    rgba[3] += pixel.a as f32 * weight;
                }
                PremulRgba8 {
                    r: rgba[0].round() as u8,
                    g: rgba[1].round() as u8,
                    b: rgba[2].round() as u8,
                    a: rgba[3].round() as u8,
                }
            }
            Self::Upscale if odd => interpolate_75_25(pixels[0], pixels[1]),
            Self::Upscale => interpolate_25_75(pixels[0], pixels[1]),
        }
    }
}

/// The positions that a pass reads along one axis to write a line.
#[derive(Debug)]
struct Taps {
    /// The positions read, in the order of the pass's input.
    input: Line,
    /// For each position written, the index in `input` of each position it
    /// reads, or `None` where it reads transparent.
    reads: Vec<Option<u32>>,
    /// For each position written, whether it is odd.
    odd: Vec<bool>,
}

impl Taps {
    /// Return what `stage` reads on a level of `size` to write `output`.
    fn new(stage: Stage<'_>, output: &Line, size: &Coord, edge_mode: EdgeMode) -> Self {
        let count = stage.taps();
        let mut wanted = Vec::new();
        // For each run written: the first position read, how many positions
        // the run reads, how many of them lie before the level, and where
        // the level starts in them.
        let mut spans = Vec::with_capacity(output.runs.len());
        for (start, len) in &output.runs {
            let first = stage.first(start);
            let last = start + i64::try_from(*len).unwrap() - 1;
            let end = stage.first(&last) + i64::try_from(count).unwrap();
            let length = (&end - &first).to_usize();
            let inside_start = (&first).max(&Coord::ZERO).clone();
            let inside_end = (&end).min(size).clone();
            let (lead, inside) = if inside_start < inside_end {
                (
                    (&inside_start - &first).to_usize(),
                    (&inside_end - &inside_start).to_usize(),
                )
            } else {
                (length, 0)
            };
            if inside > 0 {
                wanted.push((inside_start.clone(), inside));
            }
            for offset in (0..length).filter(|offset| !(lead..lead + inside).contains(offset)) {
                let position = &first + i64::try_from(offset).unwrap();
                if let Some(mapped) = extend(&position, size, edge_mode) {
                    wanted.push((mapped, 1));
                }
            }
            spans.push((first, length, lead, inside, inside_start));
        }
        let input = Line::merged(wanted);
        let total = output.len();
        let mut reads = Vec::with_capacity(total * count);
        let mut odd = Vec::with_capacity(total);
        for ((start, len), (first, length, lead, inside, inside_start)) in
            output.runs.iter().zip(spans)
        {
            let base = (inside > 0).then(|| input.index_of(&inside_start).unwrap());
            let positions: Vec<Option<u32>> = (0..length)
                .map(|offset| {
                    let index = if (lead..lead + inside).contains(&offset) {
                        Some(base.unwrap() + offset - lead)
                    } else {
                        let position = &first + i64::try_from(offset).unwrap();
                        extend(&position, size, edge_mode)
                            .map(|mapped| input.index_of(&mapped).unwrap())
                    };
                    index.map(|index| u32::try_from(index).unwrap())
                })
                .collect();
            let start_odd = start.is_odd();
            for index in 0..*len {
                let step = stage.step(start_odd, index);
                reads.extend_from_slice(&positions[step..step + count]);
                odd.push((usize::from(start_odd) + index) & 1 == 1);
            }
        }
        Self { input, reads, odd }
    }
}

/// Apply `stage` along the rows of `input`, whose columns are the positions
/// of `taps.input`.
fn pass_x(stage: Stage<'_>, taps: &Taps, input: &FilterPixmap) -> FilterPixmap {
    let count = stage.taps();
    let width = taps.odd.len();
    let input_width = input.width() as usize;
    let mut output = FilterPixmap::new(u32::try_from(width).unwrap(), input.height());
    let mut pixels = [TRANSPARENT; MAX_KERNEL_SIZE];
    for (source, target) in input
        .data()
        .chunks_exact(input_width)
        .zip(output.data_mut().chunks_exact_mut(width))
    {
        for (column, pixel) in target.iter_mut().enumerate() {
            let reads = &taps.reads[column * count..(column + 1) * count];
            for (tap, read) in pixels.iter_mut().zip(reads) {
                *tap = read.map_or(TRANSPARENT, |index| source[index as usize]);
            }
            *pixel = stage.combine(&pixels[..count], taps.odd[column]);
        }
    }
    output
}

/// Apply `stage` along the columns of `input`, whose rows are the positions
/// of `taps.input`.
fn pass_y(stage: Stage<'_>, taps: &Taps, input: &FilterPixmap) -> FilterPixmap {
    let count = stage.taps();
    let height = taps.odd.len();
    let width = input.width() as usize;
    let mut output = FilterPixmap::new(input.width(), u32::try_from(height).unwrap());
    let mut pixels = [TRANSPARENT; MAX_KERNEL_SIZE];
    let source = input.data();
    for (row, target) in output.data_mut().chunks_exact_mut(width).enumerate() {
        let reads = &taps.reads[row * count..(row + 1) * count];
        for (column, pixel) in target.iter_mut().enumerate() {
            for (tap, read) in pixels.iter_mut().zip(reads) {
                *tap = read.map_or(TRANSPARENT, |index| source[index as usize * width + column]);
            }
            *pixel = stage.combine(&pixels[..count], taps.odd[row]);
        }
    }
    output
}

/// Copy the `width` by `height` pixels of `source` at `from` to `target` at `to`.
fn copy(
    target: &mut FilterPixmap,
    to: (usize, usize),
    source: &FilterPixmap,
    from: (usize, usize),
    (width, height): (usize, usize),
) {
    let (target_width, source_width) = (target.width() as usize, source.width() as usize);
    for row in 0..height {
        let to = (to.1 + row) * target_width + to.0;
        let from = (from.1 + row) * source_width + from.0;
        target.data_mut()[to..to + width].copy_from_slice(&source.data()[from..from + width]);
    }
}

/// Return the positions of the coarser level of `coarse` positions whose
/// decimation reads `start..end` of a level of `size`.
fn decimating(
    (start, end): (&Coord, &Coord),
    size: &Coord,
    coarse: &Coord,
    edge_mode: EdgeMode,
) -> Vec<(Coord, Coord)> {
    let mut reach = Vec::new();
    // Pixel `i` reads `2i - 1..=2i + 2`.
    let first = (start - 1).shr(1).max(Coord::ZERO);
    let last = (end.shr(1) + 1).min(coarse.clone());
    if first < last {
        reach.push((first, last));
    }
    // The first pixel reads one position before the level, the last two
    // after it.
    let hits = |position: Coord| {
        extend(&position, size, edge_mode).is_some_and(|mapped| &mapped >= start && &mapped < end)
    };
    if hits(Coord::from(-1_i64)) {
        reach.push((Coord::ZERO, Coord::from(1_i64)));
    }
    if hits(size.clone()) || hits(size + 1) {
        reach.push(((coarse - 2).max(Coord::ZERO), coarse.clone()));
    }
    reach
}

/// The pixels of the coarsest level computed for earlier areas, in blocks.
#[derive(Debug, Default)]
struct Cache {
    blocks: BTreeMap<(Coord, Coord), FilterPixmap>,
    order: VecDeque<(Coord, Coord)>,
    pixels: usize,
}

/// The pyramid of levels [`apply_blur`](super::gaussian_blur::apply_blur)
/// computes on a pixmap, evaluated only where an area of the output needs it.
#[derive(Debug)]
pub(crate) struct Pyramid {
    blur: Blur,
    /// The width and height of each level, the pixmap's first.
    sizes: Vec<(Coord, Coord)>,
    /// Where each level can have content.
    content: Vec<Regions>,
    /// The largest side of a chunk of a decimated level computed at once.
    chunk: usize,
    /// The side of a block of the coarsest level in [`Self::cache`].
    block: Coord,
    cache: Cache,
}

impl Pyramid {
    /// Prepare `blur` on a pixmap of `width` by `height` whose source has
    /// content only inside `content`, in the pixmap's coordinates.
    pub(crate) fn new(blur: Blur, width: Coord, height: Coord, content: &Regions) -> Self {
        Self::with_chunk(blur, width, height, content, CHUNK)
    }

    fn with_chunk(
        blur: Blur,
        width: Coord,
        height: Coord,
        content: &Regions,
        chunk: usize,
    ) -> Self {
        let levels = blur.decimations as usize;
        let mut sizes = vec![(width, height)];
        for _ in 0..levels {
            let (width, height) = sizes.last().unwrap();
            sizes.push(((width + 1).shr(1), (height + 1).shr(1)));
        }
        let whole = Bounds::new(
            Coord::ZERO,
            Coord::ZERO,
            sizes[0].0.clone(),
            sizes[0].1.clone(),
        );
        let mut levels_content = vec![Regions::default()];
        for region in content.iter() {
            levels_content[0].insert(region.intersect(&whole));
        }
        for level in 1..sizes.len() {
            let (size, coarse) = (&sizes[level - 1], &sizes[level]);
            let mut coarser = Regions::default();
            for region in levels_content[level - 1].iter() {
                let columns =
                    decimating((&region.x0, &region.x1), &size.0, &coarse.0, blur.edge_mode);
                let rows = decimating((&region.y0, &region.y1), &size.1, &coarse.1, blur.edge_mode);
                for (x0, x1) in &columns {
                    for (y0, y1) in &rows {
                        coarser.insert(Bounds::new(x0.clone(), y0.clone(), x1.clone(), y1.clone()));
                    }
                }
            }
            levels_content.push(coarser);
        }
        // A block covers about as much of the source as the chunks that compute it.
        let block = Coord::from(((4 * chunk) >> levels.min(16)).max(16));
        Self {
            blur,
            sizes,
            content: levels_content,
            chunk,
            block,
            cache: Cache::default(),
        }
    }

    /// Return the blurred pixels of `area`, which lies in the pixmap.
    /// `source` returns the pixels of the source in a rectangle of the pixmap.
    pub(crate) fn blur(
        &mut self,
        area: &Bounds,
        source: &mut dyn FnMut(&Bounds) -> FilterPixmap,
    ) -> FilterPixmap {
        let (width, height) = area.size();
        let levels = self.blur.decimations as usize;
        let edge_mode = self.blur.edge_mode;
        let mut lines = (
            Line::single(area.x0.clone(), width as usize),
            Line::single(area.y0.clone(), height as usize),
        );
        let mut upscales = Vec::with_capacity(levels);
        for level in 1..=levels {
            let (width, height) = &self.sizes[level];
            let x = Taps::new(Stage::Upscale, &lines.0, width, edge_mode);
            let y = Taps::new(Stage::Upscale, &lines.1, height, edge_mode);
            lines = (x.input.clone(), y.input.clone());
            upscales.push((x, y));
        }
        let kernel = self.blur.kernel.clone();
        let convolve = Stage::Convolve(&kernel);
        let (width, height) = &self.sizes[levels];
        let x = Taps::new(convolve, &lines.0, width, edge_mode);
        let y = Taps::new(convolve, &lines.1, height, edge_mode);
        let decimated = if levels == 0 {
            self.decimated(0, &x.input, &y.input, source)
        } else {
            self.coarsest(&x.input, &y.input, source)
        };
        let mut pixels = pass_y(convolve, &y, &pass_x(convolve, &x, &decimated));
        for (x, y) in upscales.iter().rev() {
            pixels = pass_y(Stage::Upscale, y, &pass_x(Stage::Upscale, x, &pixels));
        }
        pixels
    }

    /// Return the pixels of the coarsest level at the positions of `columns`
    /// and `rows`, kept in blocks for later areas, which read many of them again.
    fn coarsest(
        &mut self,
        columns: &Line,
        rows: &Line,
        source: &mut dyn FnMut(&Bounds) -> FilterPixmap,
    ) -> FilterPixmap {
        let level = self.sizes.len() - 1;
        let size = Bounds::new(
            Coord::ZERO,
            Coord::ZERO,
            self.sizes[level].0.clone(),
            self.sizes[level].1.clone(),
        );
        let mut pixels = FilterPixmap::new(
            u32::try_from(columns.len()).unwrap(),
            u32::try_from(rows.len()).unwrap(),
        );
        let block = self.block.clone();
        let blocks = |start: &Coord, len: usize| {
            let first = start.div_floor(&block);
            let last = (start + i64::try_from(len).unwrap() - 1).div_floor(&block);
            let count = (last - &first).to_usize() + 1;
            (0..count).map(move |index| &first + i64::try_from(index).unwrap())
        };
        let mut x_index = 0;
        for (x_start, x_len) in &columns.runs {
            let mut y_index = 0;
            for (y_start, y_len) in &rows.runs {
                let run = Bounds::new(
                    x_start.clone(),
                    y_start.clone(),
                    x_start + i64::try_from(*x_len).unwrap(),
                    y_start + i64::try_from(*y_len).unwrap(),
                );
                for column in blocks(x_start, *x_len) {
                    for row in blocks(y_start, *y_len) {
                        let x0 = column.mul(&block);
                        let y0 = row.mul(&block);
                        let bounds =
                            Bounds::new(x0.clone(), y0.clone(), &x0 + &block, &y0 + &block)
                                .intersect(&size);
                        let part = bounds.intersect(&run);
                        if part.is_empty() || self.content[level].hull_within(&bounds).is_empty() {
                            continue;
                        }
                        let key = (column.clone(), row);
                        if !self.cache.blocks.contains_key(&key) {
                            let (width, height) = bounds.size();
                            let computed = self.decimated(
                                level,
                                &Line::single(bounds.x0.clone(), width as usize),
                                &Line::single(bounds.y0.clone(), height as usize),
                                source,
                            );
                            self.cache.pixels += computed.data().len();
                            self.cache.blocks.insert(key.clone(), computed);
                            self.cache.order.push_back(key.clone());
                            while self.cache.pixels > CACHED_PIXELS && self.cache.order.len() > 1 {
                                let oldest = self.cache.order.pop_front().unwrap();
                                let evicted = self.cache.blocks.remove(&oldest).unwrap();
                                self.cache.pixels -= evicted.data().len();
                            }
                        }
                        let (width, height) = part.size();
                        copy(
                            &mut pixels,
                            (
                                x_index + (&part.x0 - x_start).to_usize(),
                                y_index + (&part.y0 - y_start).to_usize(),
                            ),
                            &self.cache.blocks[&key],
                            ((&part.x0 - &x0).to_usize(), (&part.y0 - &y0).to_usize()),
                            (width as usize, height as usize),
                        );
                    }
                }
                y_index += y_len;
            }
            x_index += x_len;
        }
        pixels
    }

    /// Return the pixels of `level` at the positions of `columns` and `rows`.
    fn decimated(
        &self,
        level: usize,
        columns: &Line,
        rows: &Line,
        source: &mut dyn FnMut(&Bounds) -> FilterPixmap,
    ) -> FilterPixmap {
        let mut pixels = FilterPixmap::new(
            u32::try_from(columns.len()).unwrap(),
            u32::try_from(rows.len()).unwrap(),
        );
        // The source is read in whole runs, which a decimated chunk reads.
        let chunk = if level == 0 { usize::MAX } else { self.chunk };
        let row_chunks = rows.chunks(chunk);
        for (x_index, x_start, x_len) in columns.chunks(chunk) {
            for (y_index, y_start, y_len) in &row_chunks {
                let area = Bounds::new(
                    x_start.clone(),
                    y_start.clone(),
                    &x_start + i64::try_from(x_len).unwrap(),
                    y_start + i64::try_from(*y_len).unwrap(),
                );
                let hull = self.content[level].hull_within(&area);
                if hull.is_empty() {
                    continue;
                }
                let (width, height) = hull.size();
                let part = if level == 0 {
                    source(&hull)
                } else {
                    let edge_mode = self.blur.edge_mode;
                    let (fine_width, fine_height) = &self.sizes[level - 1];
                    let x = Taps::new(
                        Stage::Decimate,
                        &Line::single(hull.x0.clone(), width as usize),
                        fine_width,
                        edge_mode,
                    );
                    let y = Taps::new(
                        Stage::Decimate,
                        &Line::single(hull.y0.clone(), height as usize),
                        fine_height,
                        edge_mode,
                    );
                    let fine = self.decimated(level - 1, &x.input, &y.input, source);
                    pass_y(Stage::Decimate, &y, &pass_x(Stage::Decimate, &x, &fine))
                };
                copy(
                    &mut pixels,
                    (
                        x_index + (&hull.x0 - &x_start).to_usize(),
                        y_index + (&hull.y0 - y_start).to_usize(),
                    ),
                    &part,
                    (0, 0),
                    (width as usize, height as usize),
                );
            }
        }
        pixels
    }
}

#[cfg(test)]
mod tests {
    use super::{Blur, Line, Pyramid};
    use crate::filter::gaussian_blur::apply_blur;
    use crate::filter::pixmap::FilterPixmap;
    use crate::filter::placed::{Bounds, Coord, Regions};
    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::filter::gaussian_blur::GaussianBlur;
    use vello_common::filter_effects::EdgeMode;
    use vello_common::peniko::color::PremulRgba8;

    const EDGE_MODES: [EdgeMode; 4] = [
        EdgeMode::None,
        EdgeMode::Duplicate,
        EdgeMode::Wrap,
        EdgeMode::Mirror,
    ];

    fn blur_of(std_deviation: f32, edge_mode: EdgeMode) -> Blur {
        let blur = GaussianBlur::new(std_deviation, edge_mode);
        Blur {
            decimations: u32::try_from(blur.n_decimations).unwrap(),
            kernel: blur.kernel[..usize::from(blur.kernel_size)].to_vec(),
            edge_mode,
        }
    }

    fn whole(source: &FilterPixmap, blur: &Blur) -> FilterPixmap {
        let mut pixmap = source.clone();
        let mut scratch = FilterPixmap::new(source.width(), source.height());
        apply_blur(
            &mut pixmap,
            &mut scratch,
            blur.decimations as usize,
            &blur.kernel,
            blur.edge_mode,
        );
        pixmap
    }

    fn crop(pixmap: &FilterPixmap, area: &Bounds) -> Vec<PremulRgba8> {
        let (x0, y0) = (area.x0.to_usize(), area.y0.to_usize());
        let (width, height) = area.size();
        (y0..y0 + height as usize)
            .flat_map(|y| (x0..x0 + width as usize).map(move |x| (x, y)))
            .map(|(x, y)| pixmap.sample(x as u32, y as u32))
            .collect()
    }

    /// Return the source's pixels inside `area`, which lies in it.
    fn reader(pixmap: &FilterPixmap) -> impl FnMut(&Bounds) -> FilterPixmap + '_ {
        move |area| {
            let (width, height) = area.size();
            let mut part = FilterPixmap::new(width, height);
            for (index, pixel) in crop(pixmap, area).into_iter().enumerate() {
                part.data_mut()[index] = pixel;
            }
            part
        }
    }

    struct Random(u64);

    impl Random {
        fn next(&mut self, range: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % range
        }
    }

    #[test]
    fn lines_merge_overlapping_and_adjacent_runs() {
        let at = |start: i64, len: usize| (Coord::from(start), len);
        assert_eq!(
            Line::merged(vec![at(10, 2), at(0, 3), at(3, 1), at(11, 4), at(20, 1)]),
            Line {
                runs: vec![at(0, 4), at(10, 5), at(20, 1)]
            }
        );
    }

    #[test]
    fn areas_match_the_whole_blur_for_all_edges_sizes_and_chunks() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for round in 0..1200_u64 {
            let edge_mode = EDGE_MODES[(round % 4) as usize];
            let std_deviation = [0.4_f32, 1.0, 2.5, 4.0, 7.0, 12.0][(round / 4 % 6) as usize];
            let blur = blur_of(std_deviation, edge_mode);
            let (width, height) = (1 + random.next(40) as u32, 1 + random.next(40) as u32);
            let mut source = FilterPixmap::new(width, height);
            // Content in a few rectangles, so that most of the pixmap is empty.
            let mut content = Regions::default();
            for _ in 0..1 + random.next(3) {
                let (x0, y0) = (
                    random.next(u64::from(width)),
                    random.next(u64::from(height)),
                );
                let (x1, y1) = (
                    x0 + 1 + random.next(u64::from(width) - x0),
                    y0 + 1 + random.next(u64::from(height) - y0),
                );
                for y in y0..y1 {
                    for x in x0..x1 {
                        let alpha = random.next(256) as u8;
                        let channel = |random: &mut Random| random.next(u64::from(alpha) + 1) as u8;
                        let pixel = PremulRgba8 {
                            r: channel(&mut random),
                            g: channel(&mut random),
                            b: channel(&mut random),
                            a: alpha,
                        };
                        source.set_pixel(x as u32, y as u32, pixel);
                    }
                }
                content.insert(Bounds::new(x0 as i64, y0 as i64, x1 as i64, y1 as i64));
            }
            let expected = whole(&source, &blur);
            let chunk = [1, 2, 3, 7, 256][(round % 5) as usize];
            let mut pyramid = Pyramid::with_chunk(
                blur.clone(),
                Coord::from(width),
                Coord::from(height),
                &content,
                chunk,
            );
            for _ in 0..3 {
                let (x0, y0) = (
                    random.next(u64::from(width)),
                    random.next(u64::from(height)),
                );
                let area = Bounds::new(
                    x0 as i64,
                    y0 as i64,
                    (x0 + 1 + random.next(u64::from(width) - x0)) as i64,
                    (y0 + 1 + random.next(u64::from(height) - y0)) as i64,
                );
                let blurred = pyramid.blur(&area, &mut reader(&source));
                assert_eq!(
                    blurred.data(),
                    crop(&expected, &area),
                    "round {round}: {width}x{height}, sigma {std_deviation}, {edge_mode:?}, chunk {chunk}, area {area:?}"
                );
            }
        }
    }

    #[test]
    fn far_edges_of_huge_pixmaps_match_those_of_a_small_one() {
        // Levels of pixmaps whose sides are powers of two halve exactly, so
        // near the edges a huge pixmap blurs as a small one does if the
        // content near its edges is the same and the rest is empty.
        let small = 256_i64;
        let huge = Coord::from(1_i64).shl(70);
        for edge_mode in EDGE_MODES {
            for std_deviation in [1.5_f32, 6.0, 12.0] {
                let blur = blur_of(std_deviation, edge_mode);
                let mut source = FilterPixmap::new(small as u32, small as u32);
                for y in 0..10 {
                    for x in 0..10 {
                        source.set_pixel(x, y, PremulRgba8::from_u32(0xff_80_40_20 + x + 3 * y));
                    }
                }
                let expected = whole(&source, &blur);
                let content = Regions::of(Bounds::new(0, 0, 10, 10));
                let mut pyramid = Pyramid::new(blur.clone(), huge.clone(), huge.clone(), &content);
                let mut read = |area: &Bounds| {
                    let (width, height) = area.size();
                    let mut part = FilterPixmap::new(width, height);
                    let start = (area.x0.to_i64().unwrap(), area.y0.to_i64().unwrap());
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = source.sample(
                                (start.0 + i64::from(x)) as u32,
                                (start.1 + i64::from(y)) as u32,
                            );
                            part.set_pixel(x, y, pixel);
                        }
                    }
                    part
                };
                for (x, y) in [
                    (0_i64, 0_i64),
                    (small - 24, 0),
                    (0, small - 24),
                    (small - 24, small - 24),
                ] {
                    let local = Bounds::new(x, y, x + 24, y + 24);
                    let far = |value: i64| {
                        if value >= small / 2 {
                            &huge - (small - value)
                        } else {
                            Coord::from(value)
                        }
                    };
                    let area = Bounds::new(far(x), far(y), far(x) + 24, far(y) + 24);
                    assert_eq!(
                        pyramid.blur(&area, &mut read).data(),
                        crop(&expected, &local),
                        "{edge_mode:?}, sigma {std_deviation}, at {x}, {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn support_bounds_where_a_pixel_spreads_without_edges() {
        for std_deviation in [0.4_f32, 1.0, 2.5, 4.0, 7.0, 12.0, 30.0] {
            let blur = blur_of(std_deviation, EdgeMode::None);
            let support = blur.support().to_i64().unwrap();
            let size = 4 * support as u32 + 9;
            for offset in 0..4 {
                let mut source = FilterPixmap::new(size, 1);
                let center = 2 * support as u32 + offset;
                source.set_pixel(center, 0, PremulRgba8::from_u32(0xff_ff_ff_ff));
                let blurred = whole(&source, &blur);
                for x in 0..size {
                    if i64::from(x.abs_diff(center)) > support {
                        assert_eq!(blurred.sample(x, 0).a, 0, "sigma {std_deviation}");
                    }
                }
            }
        }
    }
}
