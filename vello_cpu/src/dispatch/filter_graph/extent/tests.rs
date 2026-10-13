// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The bounds found from geometry are those of the strips the recorder draws.

use super::{Cover, Extents, FRAME, Item, MARGIN, TILE};
use crate::dispatch::Dispatcher;
use crate::dispatch::filter_graph::Clip;
use crate::dispatch::single_threaded::SingleThreadedDispatcher;
use crate::filter::placed::{Bounds, Coord};
use crate::kurbo::{Affine, BezPath, Cap, Join, Point, Rect, Shape, Stroke};
use crate::peniko::{BlendMode, Fill};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use vello_common::color::palette::css::BLACK;
use vello_common::fearless_simd::Level;
use vello_common::paint::{Paint, PremulColor};

/// A source of pseudo-random numbers.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Return a number in `low..high`, a multiple of `1 / 2^bits` unless
    /// `bits` is `None`, where it has any fraction.
    fn number(&mut self, low: f64, high: f64, bits: Option<u32>) -> f64 {
        let unit = (self.next() >> 11) as f64 / (1_u64 << 53) as f64;
        let value = low + unit * (high - low);
        match bits {
            Some(bits) => {
                let scale = f64::from(1_u32 << bits);
                (value * scale).floor() / scale
            }
            None => value,
        }
    }

    fn pick<T: Copy>(&mut self, choices: &[T]) -> T {
        choices[(self.next() % choices.len() as u64) as usize]
    }
}

/// Return a path of a few subpaths of lines and curves inside `low..high`.
fn random_path(random: &mut Random, low: f64, high: f64, bits: Option<u32>) -> BezPath {
    let point = |random: &mut Random| {
        (
            random.number(low, high, bits),
            random.number(low, high, bits),
        )
    };
    let mut path = BezPath::new();
    for _ in 0..1 + random.next() % 3 {
        path.move_to(point(random));
        for _ in 0..1 + random.next() % 6 {
            match random.next() % 4 {
                0 | 1 => path.line_to(point(random)),
                2 => {
                    let (control, end) = (point(random), point(random));
                    path.quad_to(control, end);
                }
                _ => {
                    let (first, second, end) = (point(random), point(random), point(random));
                    path.curve_to(first, second, end);
                }
            }
        }
        if random.next().is_multiple_of(2) {
            path.close_path();
        }
    }
    path
}

/// Return a transform, which keeps coordinates multiples of the same power of
/// two if `exact` is set.
fn random_transform(random: &mut Random, exact: bool) -> Affine {
    if exact {
        return random.pick(&[Affine::IDENTITY, Affine::scale(2.0), Affine::scale(0.5)]);
    }
    match random.next() % 5 {
        0 => Affine::IDENTITY,
        1 => Affine::scale(random.number(0.5, 2.0, None)),
        2 => Affine::rotate_about(random.number(0.0, 6.3, None), Point::new(1280.0, 1280.0)),
        3 => Affine::new([1.0, 0.3, -0.2, 0.9, 10.0, -7.0]),
        _ => Affine::scale_non_uniform(random.number(0.2, 1.5, None), 1.0),
    }
}

fn random_stroke(random: &mut Random) -> Stroke {
    let mut stroke = Stroke::new(random.pick(&[0.3, 1.0, 3.5, 12.0]))
        .with_join(random.pick(&[Join::Bevel, Join::Miter, Join::Round]))
        .with_caps(random.pick(&[Cap::Butt, Cap::Square, Cap::Round]))
        .with_miter_limit(random.pick(&[1.0, 4.0, 10.0]));
    if random.next().is_multiple_of(4) {
        stroke = stroke.with_dashes(random.number(0.0, 9.0, None), [5.0, 3.0]);
    }
    stroke
}

/// Something to draw, which owns its path.
enum Drawn {
    Rect(Rect),
    Fill(BezPath, Fill, Affine),
    Stroke(BezPath, Stroke, Affine),
}

impl Drawn {
    fn item(&self) -> Item<'_> {
        match self {
            Self::Rect(rect) => Item::Rect(*rect),
            Self::Fill(path, fill_rule, transform) => Item::Fill {
                path,
                fill_rule: *fill_rule,
                transform: *transform,
            },
            Self::Stroke(path, stroke, transform) => Item::Stroke {
                path,
                stroke,
                transform: *transform,
            },
        }
    }
}

fn random_drawn(random: &mut Random, low: f64, high: f64, bits: Option<u32>, exact: bool) -> Drawn {
    match random.next() % 3 {
        0 => Drawn::Rect(Rect::new(
            random.number(low, high, bits),
            random.number(low, high, bits),
            random.number(low, high, bits),
            random.number(low, high, bits),
        )),
        1 => {
            let transform = random_transform(random, exact);
            let fill_rule = random.pick(&[Fill::NonZero, Fill::EvenOdd]);
            let path = random_path(random, low, high, bits);
            Drawn::Fill(path, fill_rule, transform)
        }
        _ => {
            let transform = random_transform(random, exact);
            let stroke = random_stroke(random);
            let path = random_path(random, low, high, bits);
            Drawn::Stroke(path, stroke, transform)
        }
    }
}

fn random_clips(random: &mut Random, low: f64, high: f64, bits: Option<u32>) -> Vec<Clip> {
    let mut clips = Vec::new();
    for _ in 0..random.next() % 3 {
        let rect = Rect::new(
            random.number(low, high, bits),
            random.number(low, high, bits),
            random.number(low, high, bits),
            random.number(low, high, bits),
        );
        clips.push(match random.next() % 4 {
            0 | 1 => Clip::Rect(rect),
            // A clip path that outlines a rectangle.
            2 => Clip::Path {
                path: rect.to_path(0.1),
                fill_rule: Fill::NonZero,
                transform: Affine::IDENTITY,
                aliasing_threshold: None,
            },
            _ => Clip::Path {
                path: random_path(random, low, high, bits),
                fill_rule: random.pick(&[Fill::NonZero, Fill::EvenOdd]),
                transform: Affine::IDENTITY,
                aliasing_threshold: None,
            },
        });
    }
    clips
}

/// Return the extents of `drawn` under `clips` and the covers of the clips.
fn found(extents: &mut Extents<'_>, drawn: &Drawn, clips: &[Clip]) -> (Bounds, Vec<Cover>) {
    let clips: Vec<&Clip> = clips.iter().collect();
    let mut covers = alloc::vec![Cover::All];
    for (index, clip) in clips.iter().enumerate() {
        let cover = extents.push_clip(&covers, &clips[..index], clip);
        covers.push(cover);
    }
    (extents.extent(&drawn.item(), &covers, &clips), covers)
}

/// Return the bounds of the strips the recorder draws for `drawn` under
/// `clips` in a viewport at the origin.
fn recorded(viewport: &Bounds, drawn: &Drawn, clips: &[Clip]) -> Bounds {
    let side = |length: Coord| u16::try_from(length.to_i64().unwrap()).unwrap();
    let size = (side(viewport.width()), side(viewport.height()));
    let paint = || Paint::Solid(PremulColor::from_alpha_color(BLACK));
    let window =
        SingleThreadedDispatcher::window(size, Level::baseline(), (0.into(), 0.into()), |target| {
            for clip in clips {
                clip.push_into(target, Affine::IDENTITY);
            }
            match drawn {
                Drawn::Rect(rect) => {
                    target.fill_rect_fast(rect, paint(), BlendMode::default(), None);
                }
                Drawn::Fill(path, fill_rule, transform) => target.fill_path(
                    path,
                    *fill_rule,
                    *transform,
                    paint(),
                    BlendMode::default(),
                    None,
                    None,
                ),
                Drawn::Stroke(path, stroke, transform) => target.stroke_path(
                    path,
                    stroke,
                    *transform,
                    paint(),
                    BlendMode::default(),
                    None,
                    None,
                ),
            }
        });
    window.window_content()
}

fn describe(drawn: &Drawn, clips: &[Clip]) -> String {
    let drawn = match drawn {
        Drawn::Rect(rect) => format!("rect {rect:?}"),
        Drawn::Fill(path, fill_rule, transform) => {
            format!("fill {fill_rule:?} {transform:?} {}", path.to_svg())
        }
        Drawn::Stroke(path, stroke, transform) => {
            format!("stroke {stroke:?} {transform:?} {}", path.to_svg())
        }
    };
    format!("{drawn} under {clips:?}")
}

#[test]
fn extents_are_those_of_the_strips_drawn_in_the_viewport() {
    // Transforms keep what is drawn inside the viewport, which the recorder
    // cuts it to.
    let viewport = Bounds::new(0, 0, 4096, 4096);
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    for case in 0..3000 {
        let bits = random.pick(&[Some(0), Some(3), None]);
        let drawn = random_drawn(&mut random, 1024.0, 1536.0, bits, false);
        let clips = random_clips(&mut random, 800.0, 3300.0, bits);
        let mut extents = Extents::new(Level::baseline(), &viewport);
        let (found, covers) = found(&mut extents, &drawn, &clips);
        let clip_refs: Vec<&Clip> = clips.iter().collect();
        let item = drawn.item();
        let drawn_bounds = recorded(&viewport, &drawn, &clips);
        assert_eq!(
            found,
            drawn_bounds,
            "case {case}: {}",
            describe(&drawn, &clips)
        );
        // Drawing in a frame gives what the recorder draws.
        let framed = extents.draw(&item, &covers, &clip_refs, &viewport);
        assert_eq!(
            framed,
            drawn_bounds,
            "case {case}: {}",
            describe(&drawn, &clips)
        );
    }
}

#[test]
fn extents_of_paths_larger_than_a_frame_are_those_of_the_strips_drawn() {
    // Coordinates that `f32` holds exactly in every frame, so that frames at
    // other origins round them alike. What is drawn is found by drawing it
    // in frames, as the test above checks against the recorder.
    let viewport = Bounds::new(0, 0, 512, 512);
    let size = 3.0 * FRAME as f64;
    let step = FRAME - 2 * MARGIN;
    let mut random = Random(0x5851_f42d_4c95_7f2d);
    for case in 0..40 {
        let drawn = random_drawn(&mut random, 0.0, size, Some(3), true);
        let clips = random_clips(&mut random, -2048.0, size + 2048.0, Some(3));
        let mut extents = Extents::new(Level::baseline(), &viewport);
        let (found, covers) = found(&mut extents, &drawn, &clips);
        let clip_refs: Vec<&Clip> = clips.iter().collect();
        let item = drawn.item();
        // Frames over all the item can draw, with a tile to spare.
        let device = item.device_bounds().inflate(TILE as f64, TILE as f64);
        let area = Bounds::covering(device).unwrap().snap_to_tiles();
        let mut drawn_bounds = Bounds::EMPTY;
        let mut y = area.y0.clone();
        while y < area.y1 {
            let mut x = area.x0.clone();
            while x < area.x1 {
                let frame =
                    Bounds::new(x.clone(), y.clone(), &x + step, &y + step).intersect(&area);
                let framed = extents.draw(&item, &covers, &clip_refs, &frame);
                drawn_bounds = drawn_bounds.union(&framed);
                x = x + step;
            }
            y = y + step;
        }
        assert_eq!(
            found,
            drawn_bounds,
            "case {case}: {}",
            describe(&drawn, &clips)
        );
    }
}
