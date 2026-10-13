// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The bounds found from geometry are those of the strips the recorder draws.

use super::{Cover, Extents, FRAME, Item, MARGIN, TILE, outside_of};
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

/// Return a coordinate in `low..high`, a multiple of `1 / 2^bits` unless
/// `bits` is `None`.
fn within(low: f64, high: f64, bits: Option<u32>) -> impl Fn(&mut Random) -> f64 {
    move |random| random.number(low, high, bits)
}

/// Return a path of a few subpaths of lines and curves with coordinates from
/// `coordinate`; one after a closed subpath starts where it does at times,
/// and a subpath is closed twice at times.
fn random_path(random: &mut Random, coordinate: &dyn Fn(&mut Random) -> f64) -> BezPath {
    let point = |random: &mut Random| (coordinate(random), coordinate(random));
    let mut path = BezPath::new();
    let mut closed = false;
    for _ in 0..1 + random.next() % 3 {
        if !closed || random.next().is_multiple_of(2) {
            path.move_to(point(random));
        }
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
        closed = random.next().is_multiple_of(2);
        if closed {
            path.close_path();
            if random.next().is_multiple_of(4) {
                path.close_path();
            }
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

/// Return a stroke, dashed with dashes and gaps of about `long` pixels at
/// times, which leave the ends of most paths in gaps.
fn random_stroke(random: &mut Random, long: f64) -> Stroke {
    let stroke = Stroke::new(random.pick(&[0.3, 1.0, 3.5, 12.0]))
        .with_join(random.pick(&[Join::Bevel, Join::Miter, Join::Round]))
        .with_caps(random.pick(&[Cap::Butt, Cap::Square, Cap::Round]))
        .with_miter_limit(random.pick(&[1.0, 4.0, 10.0]));
    match random.next() % 6 {
        0 => stroke.with_dashes(random.number(0.0, 9.0, None), [5.0, 3.0]),
        1 => {
            let dash = (random.number(0.05, 0.4, None) * long).floor().max(1.0);
            let gap = (random.number(0.5, 3.0, None) * long).floor().max(1.0);
            stroke.with_dashes(random.number(0.0, dash + gap, Some(0)), [dash, gap])
        }
        2 => {
            let dashes = [0.25, 1.5, 0.125].map(|part| (part * long).floor().max(1.0));
            stroke.with_dashes(random.number(0.0, long, Some(0)), dashes)
        }
        _ => stroke,
    }
}

/// Something to draw, which owns its path.
enum Drawn {
    Rect(Rect, Affine),
    Fill(BezPath, Fill, Affine),
    Stroke(BezPath, Stroke, Affine),
}

impl Drawn {
    fn item(&self) -> Item<'_> {
        match self {
            Self::Rect(rect, transform) => Item::Rect(*rect, Some(*transform)),
            Self::Fill(path, fill_rule, transform) => Item::Fill {
                path,
                fill_rule: *fill_rule,
                transform: *transform,
                aliasing_threshold: None,
            },
            Self::Stroke(path, stroke, transform) => Item::Stroke {
                path,
                stroke,
                transform: *transform,
                aliasing_threshold: None,
            },
        }
    }
}

/// Return something drawn with coordinates from `coordinate`, with dashes
/// about `long` pixels long at times.
fn random_drawn(
    random: &mut Random,
    coordinate: &dyn Fn(&mut Random) -> f64,
    long: f64,
    exact: bool,
) -> Drawn {
    match random.next() % 3 {
        0 => {
            let transform = if exact {
                random_transform(random, true)
            } else {
                random.pick(&[
                    Affine::IDENTITY,
                    Affine::translate((0.3, -0.7)),
                    Affine::scale_non_uniform(0.9999999, 1.0000001),
                    Affine::new([1.0 / 3.0, 0.0, 0.0, 3.0, 0.1, -1.0 / 3.0]),
                ])
            };
            let rect = Rect::new(
                coordinate(random),
                coordinate(random),
                coordinate(random),
                coordinate(random),
            );
            // The transform keeps the rectangle where it was drawn.
            let center = transform * rect.center();
            Drawn::Rect(rect, Affine::translate(rect.center() - center) * transform)
        }
        1 => {
            let transform = random_transform(random, exact);
            let fill_rule = random.pick(&[Fill::NonZero, Fill::EvenOdd]);
            let path = random_path(random, coordinate);
            Drawn::Fill(path, fill_rule, transform)
        }
        _ => {
            let transform = random_transform(random, exact);
            let stroke = random_stroke(random, long);
            let path = random_path(random, coordinate);
            Drawn::Stroke(path, stroke, transform)
        }
    }
}

fn random_clips(random: &mut Random, coordinate: &dyn Fn(&mut Random) -> f64) -> Vec<Clip> {
    let mut clips = Vec::new();
    for _ in 0..random.next() % 3 {
        let rect = Rect::new(
            coordinate(random),
            coordinate(random),
            coordinate(random),
            coordinate(random),
        );
        clips.push(match random.next() % 6 {
            0 | 1 => Clip::Rect(rect),
            // A clip path that outlines a rectangle.
            2 => Clip::Path {
                path: rect.to_path(0.1),
                fill_rule: Fill::NonZero,
                transform: Affine::IDENTITY,
                aliasing_threshold: None,
            },
            // Clip paths of lines along the axes that do not outline a
            // rectangle once.
            3 => Clip::Path {
                path: traced_back(random, &rect),
                fill_rule: random.pick(&[Fill::NonZero, Fill::EvenOdd]),
                transform: random.pick(&[Affine::IDENTITY, Affine::scale_non_uniform(1.0, 0.0)]),
                aliasing_threshold: None,
            },
            _ => Clip::Path {
                path: random_path(random, coordinate),
                fill_rule: random.pick(&[Fill::NonZero, Fill::EvenOdd]),
                transform: Affine::IDENTITY,
                aliasing_threshold: None,
            },
        });
    }
    clips
}

/// Return a path of lines along the axes over the sides of `rect` that does
/// not outline it once: sides traced back, a rectangle outlined twice or as
/// two subpaths, or one with a side of no length.
fn traced_back(random: &mut Random, rect: &Rect) -> BezPath {
    let (x0, y0, x1, y1) = (rect.x0, rect.y0, rect.x1, rect.y1);
    let mut path = BezPath::new();
    match random.next() % 4 {
        0 => {
            for x in [x0, x1] {
                path.move_to((x, y0));
                path.line_to((x, y1));
                path.close_path();
            }
        }
        1 => {
            path.move_to((x0, y0));
            for _ in 0..2 {
                path.line_to((x1, y0));
                path.line_to((x1, y1));
                path.line_to((x0, y1));
                path.line_to((x0, y0));
            }
            path.close_path();
        }
        2 => {
            path.move_to((x0, y0));
            path.line_to((x1, y0));
            path.line_to((x1, y1));
            path.close_path();
            path.move_to((x1, y1));
            path.line_to((x0, y1));
            path.line_to((x0, y0));
            path.close_path();
        }
        _ => {
            path.move_to((x0, y0));
            path.line_to((x1, y0));
            path.line_to((x1, y0));
            path.line_to((x1, y1));
            path.line_to((x0, y1));
            path.close_path();
        }
    }
    path
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
/// `clips` in `viewport`.
fn recorded(viewport: &Bounds, drawn: &Drawn, clips: &[Clip]) -> Bounds {
    let side = |length: Coord| u16::try_from(length.to_i64().unwrap()).unwrap();
    let size = (side(viewport.width()), side(viewport.height()));
    let origin = viewport.origin();
    let to_window = Affine::translate((-origin.0.to_f64(), -origin.1.to_f64()));
    let paint = || Paint::Solid(PremulColor::from_alpha_color(BLACK));
    let window = SingleThreadedDispatcher::window(size, Level::baseline(), origin, |target| {
        for clip in clips {
            clip.push_into(target, to_window);
        }
        match drawn {
            Drawn::Rect(rect, transform) => target.fill_transformed_rect(
                rect,
                to_window * *transform,
                paint(),
                BlendMode::default(),
                None,
            ),
            Drawn::Fill(path, fill_rule, transform) => target.fill_path(
                path,
                *fill_rule,
                to_window * *transform,
                paint(),
                BlendMode::default(),
                None,
                None,
            ),
            Drawn::Stroke(path, stroke, transform) => target.stroke_path(
                path,
                stroke,
                to_window * *transform,
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
        Drawn::Rect(rect, transform) => format!("rect {rect:?} {transform:?}"),
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
        let drawn = random_drawn(&mut random, &within(1024.0, 1536.0, bits), 512.0, false);
        let clips = random_clips(&mut random, &within(800.0, 3300.0, bits));
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

/// Dashed strokes across most of the viewport, which are drawn in pieces of it
/// around the dashes that reach each, see `Extents::far_path`.
#[test]
fn extents_of_dashed_strokes_across_the_viewport_are_those_drawn() {
    let viewport = Bounds::new(0, 0, 4096, 4096);
    let mut random = Random(0x6a09_e667_f3bc_c909);
    let mut cases = 0;
    while cases < 600 {
        let bits = random.pick(&[Some(0), Some(3), None]);
        let coordinate = within(100.0, 4000.0, bits);
        let long = random.pick(&[8.0, 64.0, 512.0, 2048.0]);
        let stroke = random_stroke(&mut random, long);
        if stroke.dash_pattern.is_empty() {
            continue;
        }
        cases += 1;
        let transform = random.pick(&[Affine::IDENTITY, Affine::scale(0.5)]);
        let drawn = Drawn::Stroke(random_path(&mut random, &coordinate), stroke, transform);
        let clips = random_clips(&mut random, &within(-100.0, 4200.0, bits));
        let mut extents = Extents::new(Level::baseline(), &viewport);
        let (found, _) = found(&mut extents, &drawn, &clips);
        assert_eq!(
            found,
            recorded(&viewport, &drawn, &clips),
            "case {cases}: {}",
            describe(&drawn, &clips)
        );
    }
}

#[test]
fn extents_near_the_viewport_origin_are_those_the_recorder_draws() {
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    for case in 0..3000 {
        // Paddings are whole tiles, see `FilterData::source_padding`.
        let (x, y) = random.pick(&[(0, 0), (-4, -8), (-64, -64), (-1000, -76), (-60_000, -4)]);
        let viewport = Bounds::new(x, y, x + 300, y + 200);
        // Near the corners of the first tiles, inside and outside the
        // viewport, or of any tile of it, a little to either side.
        let near = move |random: &mut Random, origin: i64| {
            let tiles = random.pick(&[12, 50]);
            let corner = origin + TILE * (random.next() % tiles) as i64 - TILE;
            let off = random.pick(&[
                0.0,
                2_f64.powi(-22),
                -(2_f64.powi(-22)),
                2_f64.powi(-30),
                -(2_f64.powi(-30)),
                0.5,
                -1e-7,
                3.999_999,
            ]);
            corner as f64 + off
        };
        // Alternate between the axes, so that points lie near corners.
        let axis = core::cell::Cell::new(0);
        let coordinate = |random: &mut Random| {
            axis.set(axis.get() + 1);
            if axis.get() % 2 == 1 {
                near(random, x)
            } else {
                near(random, y)
            }
        };
        let drawn = random_drawn(&mut random, &coordinate, 8.0, false);
        let clips = random_clips(&mut random, &coordinate);
        let mut extents = Extents::new(Level::baseline(), &viewport);
        let (found, covers) = found(&mut extents, &drawn, &clips);
        // What is found outside the viewport is checked by the tests below.
        let clip_refs: Vec<&Clip> = clips.iter().collect();
        let home = extents.home_extent(&drawn.item(), &covers, &clip_refs);
        assert!(found.contains(&home));
        assert_eq!(
            home,
            recorded(&viewport, &drawn, &clips),
            "case {case}: {}",
            describe(&drawn, &clips)
        );
    }
}

/// Return the bounds of the strips drawn for `drawn` under `clips`, by
/// drawing them in `viewport` as the recorder does and outside of it in
/// frames, which round coordinates that `f32` holds exactly in every frame
/// alike, as the tests above check against the recorder; and the bounds
/// found for them.
fn drawn_in_frames(viewport: &Bounds, drawn: &Drawn, clips: &[Clip]) -> (Bounds, Bounds) {
    let step = FRAME - 2 * MARGIN;
    let mut extents = Extents::new(Level::baseline(), viewport);
    let (found, covers) = found(&mut extents, drawn, clips);
    let clip_refs: Vec<&Clip> = clips.iter().collect();
    let item = drawn.item();
    // Frames over all the item can draw, with a tile to spare.
    let device = item.device_bounds().inflate(TILE as f64, TILE as f64);
    let area = Bounds::covering(device).unwrap().snap_to_tiles();
    let mut drawn_bounds = recorded(viewport, drawn, clips);
    for piece in outside_of(&area, &viewport.snap_to_tiles()) {
        let mut y = piece.y0.clone();
        while y < piece.y1 {
            let mut x = piece.x0.clone();
            while x < piece.x1 {
                let frame =
                    Bounds::new(x.clone(), y.clone(), &x + step, &y + step).intersect(&piece);
                let framed = extents.draw(&item, &covers, &clip_refs, &frame);
                drawn_bounds = drawn_bounds.union(&framed);
                x = x + step;
            }
            y = y + step;
        }
    }
    (drawn_bounds, found)
}

#[test]
fn extents_of_paths_larger_than_a_frame_are_those_of_the_strips_drawn() {
    let viewport = Bounds::new(0, 0, 512, 512);
    let size = 3.0 * FRAME as f64;
    let mut random = Random(0x5851_f42d_4c95_7f2d);
    for case in 0..60 {
        let drawn = random_drawn(&mut random, &within(0.0, size, Some(3)), size, true);
        let clips = random_clips(&mut random, &within(-2048.0, size + 2048.0, Some(3)));
        let (drawn_bounds, found) = drawn_in_frames(&viewport, &drawn, &clips);
        assert_eq!(
            found,
            drawn_bounds,
            "case {case}: {}",
            describe(&drawn, &clips)
        );
    }
}

fn line(from: (f64, f64), to: (f64, f64)) -> BezPath {
    let mut path = BezPath::new();
    path.move_to(from);
    path.line_to(to);
    path
}

#[test]
fn dashes_between_gaps_longer_than_a_frame_are_found() {
    let viewport = Bounds::new(0, 0, 512, 512);
    let cases = [
        // The one dash lies far from the ends and from the viewport.
        (
            line((-500_000.0, 100_000.0), (500_000.0, 100_000.0)),
            Stroke::new(1.0).with_dashes(500_000.0, [10.0, 999_990.0]),
        ),
        (
            line((100_032.0, -500_000.0), (100_032.0, 500_000.0)),
            Stroke::new(2.0).with_dashes(500_000.0, [100_000.0, 900_000.0]),
        ),
        // The last line starts the pattern over from where the closed
        // subpath starts, and draws its one dash far from its ends.
        (
            {
                let mut path = line((-200_000.0, 100_000.0), (50_000.0, 100_000.0));
                path.close_path();
                path.line_to((800_000.0, 100_000.0));
                path
            },
            Stroke::new(1.0).with_dashes(500_000.0, [10.0, 999_990.0]),
        ),
        // Dashes start and end inside a curve.
        (
            {
                let mut path = BezPath::new();
                path.move_to((-300_000.0, 200_000.0));
                path.curve_to(
                    (-100_000.0, -400_000.0),
                    (100_000.0, 800_000.0),
                    (300_000.0, 200_000.0),
                );
                path
            },
            Stroke::new(3.0).with_dashes(0.0, [70_000.0, 400_000.0]),
        ),
        // A curve that starts in a gap, whose end is the start of the only dash
        // beyond.
        (
            {
                let mut path = BezPath::new();
                path.move_to((0.0, 0.0));
                path.quad_to((200_000.0, -400_000.0), (400_000.0, 0.0));
                path
            },
            Stroke::new(1.0).with_dashes(9_950_020.0, [10.0, 50_000.0, 10.0, 10_000_000.0]),
        ),
    ];
    for (path, stroke) in cases {
        let drawn = Drawn::Stroke(path, stroke, Affine::IDENTITY);
        let (drawn_bounds, found) = drawn_in_frames(&viewport, &drawn, &[]);
        assert!(!drawn_bounds.is_empty());
        assert_eq!(found, drawn_bounds, "{}", describe(&drawn, &[]));
    }
}

#[test]
fn clip_paths_of_lines_traced_back_cover_only_their_lines() {
    let viewport = Bounds::new(0, 0, 512, 512);
    let mut clip = BezPath::new();
    for x in [100_100.0, 100_200.0] {
        clip.move_to((x, 100.0));
        clip.line_to((x, 200.0));
        clip.close_path();
    }
    for fill_rule in [Fill::NonZero, Fill::EvenOdd] {
        let clips = [Clip::Path {
            path: clip.clone(),
            fill_rule,
            transform: Affine::IDENTITY,
            aliasing_threshold: None,
        }];
        // Inside the clip's bounds, between its lines.
        let square = Rect::new(100_140.0, 140.0, 100_160.0, 160.0);
        for drawn in [
            Drawn::Rect(square, Affine::IDENTITY),
            Drawn::Fill(square.to_path(0.1), Fill::NonZero, Affine::IDENTITY),
        ] {
            let (drawn_bounds, found) = drawn_in_frames(&viewport, &drawn, &clips);
            assert_eq!(found, drawn_bounds, "{}", describe(&drawn, &clips));
        }
    }
}

#[test]
fn lines_left_of_the_clips_are_found_along_their_left_edge() {
    let viewport = Bounds::new(0, 0, 512, 512);
    let mut triangle = BezPath::new();
    triangle.move_to((0.0, 0.0));
    triangle.line_to((150_000.0, 50_000.0));
    triangle.line_to((0.0, 100_000.0));
    triangle.close_path();
    // A clip path left of the clip rectangle, whose lines are longer than
    // `fill_clip` draws whole.
    let clips = [
        Clip::Rect(Rect::new(200_001.0, 10_000.0, 210_000.0, 90_000.0)),
        Clip::Path {
            path: triangle,
            fill_rule: Fill::NonZero,
            transform: Affine::IDENTITY,
            aliasing_threshold: None,
        },
    ];
    let square = Rect::new(0.0, 0.0, 300_000.0, 100_000.0).to_path(0.1);
    let drawn = Drawn::Fill(square, Fill::NonZero, Affine::IDENTITY);
    let (drawn_bounds, found) = drawn_in_frames(&viewport, &drawn, &clips);
    assert!(!drawn_bounds.is_empty());
    assert_eq!(found, drawn_bounds, "{}", describe(&drawn, &clips));
}

#[test]
fn rectangles_round_as_the_recorder_maps_them() {
    let viewport = Bounds::new(-64, -64, 236, 136);
    // Moved by 0.3, then into the viewport by 64, its left side lies at 88
    // in `f32`; moved into the viewport by 64.3, just left of it.
    let drawn = Drawn::Rect(
        Rect::new(23.699_996_185_302_727, 0.0, 100.0, 50.0),
        Affine::translate((0.3, 0.0)),
    );
    let mut extents = Extents::new(Level::baseline(), &viewport);
    let (found, _) = found(&mut extents, &drawn, &[]);
    assert_eq!(found, recorded(&viewport, &drawn, &[]));
    assert_eq!(found.x0, Coord::from(20));
}

#[test]
fn rectangles_without_bounds_are_cut_as_the_recorder_cuts_them() {
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let rects = [
        Rect::new(-inf, 10.0, 30.0, 40.0),
        Rect::new(10.0, -inf, 40.0, inf),
        Rect::new(-inf, -inf, inf, inf),
        Rect::new(inf, 10.0, -inf, 40.0),
        Rect::new(10.0, 20.0, inf, 20.0),
        Rect::new(nan, 10.0, 30.0, 40.0),
    ];
    let transforms = [
        Affine::IDENTITY,
        Affine::translate((0.3, -0.7)),
        Affine::new([0.0, 2.0, -1.0, 0.0, 5.0, 3.0]),
    ];
    for viewport in [
        Bounds::new(0, 0, 256, 128),
        Bounds::new(-1000, -76, -744, 52),
    ] {
        let origin = viewport.origin();
        let to_viewport = Affine::translate((origin.0.to_f64(), origin.1.to_f64()));
        // Clips inside the viewport and reaching out of it, where the viewport
        // cuts a rectangle without bounds as the recorder does: a rectangle, a
        // path that is one, which covers whole tiles, the path turned, and a
        // rectangle under the path.
        let path = |clip: Rect, turn: f64| Clip::Path {
            path: clip.to_path(0.1),
            fill_rule: Fill::NonZero,
            transform: to_viewport * Affine::rotate(turn) * to_viewport.inverse(),
            aliasing_threshold: None,
        };
        let mut clips = alloc::vec![alloc::vec![]];
        for clip in [
            Rect::new(5.0, 15.0, 90.0, 70.0),
            Rect::new(-60.0, -2000.0, 90.0, 70.0),
        ] {
            let clip = to_viewport.transform_rect_bbox(clip);
            clips.extend([
                alloc::vec![Clip::Rect(clip)],
                alloc::vec![path(clip, 0.0)],
                alloc::vec![path(clip, 0.1)],
                alloc::vec![path(clip, 0.0), Clip::Rect(clip.inflate(-2.0, -2.0))],
            ]);
        }
        for rect in rects {
            for transform in transforms {
                for clips in &clips {
                    let drawn = Drawn::Rect(rect, to_viewport * transform);
                    let mut extents = Extents::new(Level::baseline(), &viewport);
                    let (found, _) = found(&mut extents, &drawn, clips);
                    assert_eq!(
                        found,
                        recorded(&viewport, &drawn, clips),
                        "{viewport:?}: {}",
                        describe(&drawn, clips)
                    );
                }
            }
        }
    }
}
