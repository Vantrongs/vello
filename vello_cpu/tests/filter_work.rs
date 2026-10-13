// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory a filter layer allocates follows the visible view, not how far
//! its source lies, how wide its blur is or how deeply filters are nested.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use vello_cpu::color::AlphaColor;
use vello_cpu::color::palette::css::BLUE;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, BezPath, Rect, Shape};
use vello_cpu::peniko::Fill;
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
};

/// Counts the bytes each thread allocates, and refuses any allocation past
/// [`LIVE_LIMIT`] or [`TOTAL_LIMIT`], which aborts the test instead of
/// exhausting the machine's memory.
struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const LIVE_LIMIT: isize = 1 << 30;
const TOTAL_LIMIT: usize = 16 << 30;

#[derive(Clone, Copy, Debug, Default)]
struct Usage {
    /// The bytes allocated less those freed on this thread, which is negative
    /// when it frees memory another thread allocated.
    live: isize,
    /// The most bytes live at once.
    peak: isize,
    /// The bytes allocated in total.
    total: usize,
}

thread_local! {
    static USAGE: Cell<Usage> = const {
        Cell::new(Usage {
            live: 0,
            peak: 0,
            total: 0,
        })
    };
}

/// Count an allocation of `size` bytes, or return `false` if it is refused.
fn reserve(size: usize) -> bool {
    USAGE
        .try_with(|usage| {
            let mut current = usage.get();
            current.live = current.live.wrapping_add_unsigned(size);
            current.total = current.total.saturating_add(size);
            if current.live > LIVE_LIMIT || current.total > TOTAL_LIMIT {
                return false;
            }
            current.peak = current.peak.max(current.live);
            usage.set(current);
            true
        })
        .unwrap_or(true)
}

fn release(size: usize) {
    let _ = USAGE.try_with(|usage| {
        let mut current = usage.get();
        current.live = current.live.wrapping_sub_unsigned(size);
        usage.set(current);
    });
}

// SAFETY: Every call is forwarded to `System`, which upholds the contract;
// refusing an allocation returns null as the contract allows.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !reserve(layout.size()) {
            return core::ptr::null_mut();
        }
        // SAFETY: Forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !reserve(layout.size()) {
            return core::ptr::null_mut();
        }
        // SAFETY: Forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        release(layout.size());
        // SAFETY: Forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !reserve(new_size) {
            return core::ptr::null_mut();
        }
        release(layout.size());
        // SAFETY: Forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Return the memory `work` allocates on this thread: the most it holds at
/// once beyond what was live before, and the total.
fn measure<T>(work: impl FnOnce() -> T) -> (Usage, T) {
    let before = USAGE.with(|usage| {
        let mut current = usage.get();
        current.peak = current.live;
        current.total = 0;
        usage.set(current);
        current
    });
    let result = work();
    let after = USAGE.with(Cell::get);
    let usage = Usage {
        live: after.live - before.live,
        peak: after.peak - before.live,
        total: after.total,
    };
    (usage, result)
}

fn context(width: u16, height: u16) -> RenderContext {
    RenderContext::new_with(
        width,
        height,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    )
}

/// Flush and render `context`, and return its pixels.
fn render(mut context: RenderContext) -> Vec<u8> {
    context.flush();
    let mut pixels = Pixmap::new(context.width(), context.height());
    context.render_with(
        &mut pixels,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: RenderMode::OptimizeSpeed,
            ..Default::default()
        },
    );
    pixels.data_as_u8_slice().to_vec()
}

/// Render a square shadowed `depth` times by drop shadows that each move it
/// 65536 pixels right, so that its last shadow lands in a 64 by 64 view. Every
/// shadow keeps its source, so each nested layer reads one more copy.
fn nested_shadows(depth: u32) -> Vec<u8> {
    const STEP: f32 = 65_536.0;
    let mut context = context(64, 64);
    for _ in 0..depth {
        context.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadow {
            dx: STEP,
            dy: 0.0,
            std_deviation: 1.0,
            color: AlphaColor::from_rgba8(0, 0, 255, 255),
            edge_mode: EdgeMode::None,
        }));
    }
    context.set_paint(BLUE);
    let x = 16.0 - f64::from(STEP) * f64::from(depth);
    context.fill_rect(&Rect::new(x, 16.0, x + 32.0, 48.0));
    for _ in 0..depth {
        context.pop_layer();
    }
    render(context)
}

#[test]
fn nested_shadows_allocate_quadratically_in_depth() {
    let usage: Vec<Usage> = [4, 8, 16]
        .into_iter()
        .map(|depth| {
            let (usage, pixels) = measure(|| nested_shadows(depth));
            let center = (32 * 64 + 32) * 4;
            assert_eq!(
                &pixels[center..center + 4],
                [0, 0, 255, 255],
                "depth {depth}"
            );
            usage
        })
        .collect();
    // Each doubling of the depth at most quadruples the source regions read.
    for pair in usage.windows(2) {
        assert!(pair[1].total <= 5 * pair[0].total, "{usage:?}");
        assert!(pair[1].peak <= 5 * pair[0].peak, "{usage:?}");
    }
}

/// Render a square shadowed `depth` times by drop shadows that each keep their
/// source and move it by a distinct power of two times 128 pixels, alternately
/// right and left, so that it has `2^depth` copies, of which only the one
/// every shadow moves lies in a 64 by 64 view.
fn spread_shadows(depth: u32) -> Vec<u8> {
    let offset = |level: u32| {
        let distance = 128.0 * 2_f32.powi(i32::try_from(level).unwrap());
        if level.is_multiple_of(2) {
            distance
        } else {
            -distance
        }
    };
    let mut context = context(64, 64);
    for level in (0..depth).rev() {
        context.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadow {
            dx: offset(level),
            dy: 0.0,
            std_deviation: 0.0,
            color: AlphaColor::from_rgba8(0, 0, 255, 255),
            edge_mode: EdgeMode::None,
        }));
    }
    context.set_paint(BLUE);
    let x = 16.0
        - (0..depth)
            .map(|level| f64::from(offset(level)))
            .sum::<f64>();
    context.fill_rect(&Rect::new(x, 16.0, x + 32.0, 48.0));
    for _ in 0..depth {
        context.pop_layer();
    }
    render(context)
}

#[test]
fn shadows_moving_by_distinct_distances_allocate_linearly_in_depth() {
    let mut previous: Option<(u32, Usage)> = None;
    for depth in [8, 16, 24] {
        let (usage, pixels) = measure(|| spread_shadows(depth));
        for (index, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
            let (x, y) = (index % 64, index / 64);
            let inside = (16..48).contains(&x) && (16..48).contains(&y);
            let expected = if inside { [0, 0, 255, 255] } else { [0; 4] };
            assert_eq!(*pixel, expected, "depth {depth} at ({x}, {y})");
        }
        // Twice the layers allocate at most about twice as much.
        if let Some((shallow, before)) = previous {
            assert!(
                usage.total <= 3 * before.total && usage.peak <= 3 * before.peak,
                "depth {depth}: {usage:?}, at {shallow}: {before:?}"
            );
        }
        previous = Some((depth, usage));
    }
}

/// Render, in a 64 by 64 view, a source that lies `distance` pixels right of
/// and below it, moved into view by an offset: a band reaching back to the
/// origin and a blurred square nested in the layer.
fn moved_into_view(distance: f32) -> Vec<u8> {
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: -distance,
        dy: -distance,
    }));
    let distance = f64::from(distance);
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(
        distance + 4.0,
        0.0,
        distance + 12.0,
        distance + 64.0,
    ));
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    }));
    context.fill_rect(&Rect::new(
        distance + 24.0,
        distance + 24.0,
        distance + 48.0,
        distance + 48.0,
    ));
    context.pop_layer();
    context.pop_layer();
    render(context)
}

#[test]
fn offset_allocates_the_same_however_far_the_source_lies() {
    // The first distance keeps the source inside the `u16` domain, the later
    // ones lie in it, at its end and beyond it.
    let (near, expected) = measure(|| moved_into_view(96.0));
    assert!(
        expected
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[3] != 0)
    );
    for distance in [4_096.0, 60_000.0, 262_144.0, 16_777_216.0] {
        let (usage, pixels) = measure(|| moved_into_view(distance));
        assert!(pixels == expected, "pixels at distance {distance}");
        assert!(
            usage.peak <= near.peak + near.peak / 4 && usage.total <= near.total + near.total / 4,
            "distance {distance}: {usage:?}, at 96: {near:?}"
        );
    }
}

/// Blur, in a 64 by 64 view, a band reaching `length` pixels down from its
/// top, whose end the blur's edges depend on, drawn as a rectangle or, if
/// `as_path` is set, as a path.
fn blurred_band(length: f64, edge_mode: EdgeMode, as_path: bool) -> Vec<u8> {
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode,
    }));
    context.set_paint(BLUE);
    let band = Rect::new(20.5, 8.0, 40.5, length);
    if as_path {
        context.fill_path(&band.to_path(0.1));
    } else {
        context.fill_rect(&band);
    }
    context.pop_layer();
    render(context)
}

#[test]
fn blur_of_a_band_costs_the_same_however_long_it_is() {
    for edge_mode in [EdgeMode::None, EdgeMode::Duplicate] {
        for as_path in [false, true] {
            let (short, expected) = measure(|| blurred_band(4_096.0, edge_mode, as_path));
            for length in [65_536.0, 1_048_576.0, 16_777_216.0] {
                let (usage, pixels) = measure(|| blurred_band(length, edge_mode, as_path));
                let case = format!("{edge_mode:?}, as path {as_path}, at length {length}");
                assert!(pixels == expected, "{case}");
                assert!(
                    usage.peak <= short.peak + short.peak / 4
                        && usage.total <= short.total + short.total / 4,
                    "{case}: {usage:?}, at 4096: {short:?}"
                );
            }
        }
    }
}

/// Blur, in a 64 by 64 view, a band reaching `band` pixels down from its top
/// under a clip path reaching `clip` pixels down, whose sides lean by a pixel
/// over that length, so that the band and the clip cover tiles together all
/// along the shorter of them.
fn blurred_band_under_a_leaning_clip(band: f64, clip: f64, edge_mode: EdgeMode) -> Vec<u8> {
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode,
    }));
    let mut path = BezPath::new();
    path.move_to((20.0, 0.0));
    path.line_to((40.0, 0.0));
    path.line_to((41.0, clip));
    path.line_to((21.0, clip));
    path.close_path();
    context.push_clip_path(&path);
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(20.0, 0.0, 42.0, band));
    context.pop_clip();
    context.pop_layer();
    render(context)
}

#[test]
fn blur_of_a_band_under_a_leaning_clip_costs_the_same_however_long_it_is() {
    for edge_mode in [EdgeMode::None, EdgeMode::Duplicate] {
        let (short, _) = measure(|| blurred_band_under_a_leaning_clip(4_096.0, 4_096.0, edge_mode));
        for length in [262_144.0, 1_048_576.0, 16_777_216.0] {
            let (usage, pixels) =
                measure(|| blurred_band_under_a_leaning_clip(length, length, edge_mode));
            // The view and the blur's reach see the same of a shorter band.
            let expected = blurred_band_under_a_leaning_clip(4_096.0, length, edge_mode);
            let case = format!("{edge_mode:?} at length {length}");
            assert!(pixels == expected, "{case}");
            assert!(
                usage.peak <= short.peak + short.peak / 4
                    && usage.total <= short.total + short.total / 4,
                "{case}: {usage:?}, at 4096: {short:?}"
            );
        }
    }
}

/// Blur, in a 64 by 64 view, a fill with a hole leaning right by a pixel
/// every 64 pixels down its `length`, under a clip path that runs inside the
/// hole 6 pixels from its sides: the fill and the clip cover no tile
/// together, which only drawing them tells, as the fill covers tiles on both
/// sides of the clip over any 384 pixels of its length.
fn blurred_fill_around_a_leaning_clip(length: f64) -> Vec<u8> {
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    }));
    let lean = length / 64.0;
    let band = |path: &mut BezPath, left: f64, right: f64| {
        path.move_to((left, 0.0));
        path.line_to((right, 0.0));
        path.line_to((right + lean, length));
        path.line_to((left + lean, length));
        path.close_path();
    };
    let mut clip = BezPath::new();
    band(&mut clip, 26.0, 34.0);
    context.push_clip_path(&clip);
    let mut fill = Rect::new(-100.0, 0.0, lean + 200.0, length).to_path(0.1);
    band(&mut fill, 20.0, 40.0);
    context.set_fill_rule(Fill::EvenOdd);
    context.set_paint(BLUE);
    context.fill_path(&fill);
    context.pop_clip();
    context.pop_layer();
    render(context)
}

#[test]
fn a_fill_around_a_clip_holds_the_same_memory_however_long_they_are() {
    let (short, expected) = measure(|| blurred_fill_around_a_leaning_clip(262_144.0));
    assert!(expected.iter().all(|value| *value == 0));
    let mut totals = vec![short.total];
    for length in [1_048_576.0, 4_194_304.0] {
        let (usage, pixels) = measure(|| blurred_fill_around_a_leaning_clip(length));
        let case = format!("at length {length}");
        assert!(pixels == expected, "{case}");
        assert!(
            usage.peak <= short.peak + short.peak / 4,
            "{case}: {usage:?}, at 262144: {short:?}"
        );
        totals.push(usage.total);
    }
    // Every part along the hole is drawn, which a total four times as large
    // for each length four times as long counts.
    for pair in totals.windows(2) {
        assert!(pair[1] <= 5 * pair[0], "{totals:?}");
    }
}

/// Blur, in a 64 by 64 view, a square and a path that only moves, from the
/// origin to `distance` pixels down and right of it, and so draws nothing.
fn blurred_with_moves(distance: f64, edge_mode: EdgeMode) -> Vec<u8> {
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode,
    }));
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(20.0, 20.0, 44.0, 44.0));
    let mut moves = BezPath::new();
    moves.move_to((0.0, 0.0));
    moves.move_to((distance, distance));
    context.fill_path(&moves);
    context.pop_layer();
    render(context)
}

#[test]
fn moves_that_draw_nothing_cost_nothing_however_far_they_reach() {
    for edge_mode in [EdgeMode::None, EdgeMode::Duplicate] {
        let (near, expected) = measure(|| blurred_with_moves(64.0, edge_mode));
        for distance in [16_384.0, 16_777_216.0, 1e15] {
            let (usage, pixels) = measure(|| blurred_with_moves(distance, edge_mode));
            assert!(pixels == expected, "{edge_mode:?} at distance {distance}");
            assert!(
                usage.peak <= near.peak + near.peak / 4
                    && usage.total <= near.total + near.total / 4,
                "{edge_mode:?} at distance {distance}: {usage:?}, at 64: {near:?}"
            );
        }
    }
}

/// Blur, in a 12 by 12 view, a square centered on it with sides `size` pixels
/// long, drawn and blurred in a space scaled by `scale`.
fn blurred_square(std_deviation: f32, size: f64, scale: f64) -> Vec<u8> {
    let mut context = context(12, 12);
    context.set_transform(Affine::scale(scale));
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    }));
    context.set_paint(BLUE);
    let (center, half) = (6.0 / scale, size / 2.0 / scale);
    context.fill_rect(&Rect::new(
        center - half,
        center - half,
        center + half,
        center + half,
    ));
    context.pop_layer();
    render(context)
}

#[test]
fn blur_of_a_square_covering_its_reach_is_opaque() {
    for scale in [1.0, 4.0] {
        for std_deviation in [4.0, 40.0, 400.0] {
            // The blur reads three standard deviations around each pixel,
            // which hold 99.7% of its weight; the square reaches four.
            let size = 12.0 + 8.0 * f64::from(std_deviation) * scale;
            let (usage, pixels) = measure(|| blurred_square(std_deviation, size, scale));
            for pixel in pixels.as_chunks::<4>().0 {
                assert!(
                    pixel[0] == 0 && pixel[1] == 0 && pixel[2] >= 253 && pixel[3] >= 253,
                    "sigma {std_deviation} at scale {scale}: {pixel:?}"
                );
            }
            assert!(
                usage.peak <= 32 << 20,
                "sigma {std_deviation} at scale {scale}: {usage:?}"
            );
        }
    }
}

#[test]
fn blur_holds_the_same_memory_however_wide_it_is() {
    // A blur reads every pixel of its source within its reach, which takes
    // work growing with the square of its width until it holds the whole
    // source, but holds no more than a tile of it at once. At larger scales,
    // the largest standard deviations exceed `f32` in the target's space.
    let (narrow, _) = measure(|| blurred_square(400.0, 2048.0, 1.0));
    for scale in [1.0, 4.0, 1.0e30] {
        for std_deviation in [400.0, 4_000.0, 1.0e9, 1.0e30, f32::MAX] {
            let (usage, pixels) = measure(|| blurred_square(std_deviation, 2048.0, scale));
            // The view lies at the center of the square, where its blur keeps
            // the share of a Gaussian that lies within the square, less what
            // decimating the source changes.
            let share = erf(1024.0 / (f64::from(std_deviation) * scale * 2.0_f64.sqrt()));
            let alpha = share * share * 255.0;
            for pixel in pixels.as_chunks::<4>().0 {
                assert!(
                    (f64::from(pixel[3]) - alpha).abs() <= 2.0,
                    "sigma {std_deviation} at scale {scale}: {pixel:?}, expected {alpha}"
                );
            }
            assert!(
                usage.peak <= narrow.peak + narrow.peak / 4,
                "sigma {std_deviation} at scale {scale}: {usage:?}, at 400: {narrow:?}"
            );
        }
    }
}

/// Render a square in a 64 by 64 view under `primitive`, whose moved copy lands
/// `distance` times `scale` pixels right of and below it, far outside the view.
fn moved_far(primitive: impl Fn(f32) -> FilterPrimitive, distance: f32, scale: f64) -> Vec<u8> {
    let mut context = context(64, 64);
    context.set_transform(Affine::scale(scale));
    context.push_filter_layer(Filter::from_primitive(primitive(distance)));
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0).scale_from_origin(1.0 / scale));
    context.pop_layer();
    render(context)
}

#[test]
fn copies_moved_past_any_coordinate_domain_leave_the_view_as_it_is() {
    let shadow = |std_deviation| {
        move |distance| FilterPrimitive::DropShadow {
            dx: distance,
            dy: distance,
            std_deviation,
            color: AlphaColor::from_rgba8(255, 0, 0, 255),
            edge_mode: EdgeMode::None,
        }
    };
    let offset = |distance| FilterPrimitive::Offset {
        dx: distance,
        dy: distance,
    };
    let (near, source) = measure(|| moved_far(shadow(0.0), 64.0, 1.0));
    let mut square = context(64, 64);
    square.set_paint(BLUE);
    square.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
    assert!(source == render(square));
    // At larger scales, the largest distance exceeds `f32` in the target's space.
    for (distance, scale) in [
        (3.2e9, 1.0),
        (1.0e19, 1.0),
        (f32::MAX, 1.0),
        (f32::MAX, 4.0),
        (f32::MAX, 1.0e30),
    ] {
        for std_deviation in [0.0, 2.0] {
            let (usage, pixels) = measure(|| moved_far(shadow(std_deviation), distance, scale));
            assert!(
                pixels == source,
                "shadow {std_deviation} at {distance}, {scale}"
            );
            assert!(
                usage.peak <= 2 * near.peak && usage.total <= 2 * near.total,
                "shadow {std_deviation} at {distance}, {scale}: {usage:?}, near: {near:?}"
            );
        }
        let pixels = moved_far(offset, distance, scale);
        assert!(
            pixels.iter().all(|&value| value == 0),
            "offset {distance}, {scale}"
        );
    }
}

/// Render a square in a 64 by 64 view inside `filters`, each nested in the one
/// before, under a transform scaling by `scale`.
fn nested_in(filters: &[FilterPrimitive], scale: f64) -> Vec<u8> {
    let mut context = context(64, 64);
    context.set_transform(Affine::scale(scale));
    for filter in filters {
        context.push_filter_layer(Filter::from_primitive(filter.clone()));
    }
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0).scale_from_origin(1.0 / scale));
    for _ in filters {
        context.pop_layer();
    }
    render(context)
}

#[test]
fn opposite_shifts_past_any_coordinate_domain_cancel() {
    let offset = |distance: f32| FilterPrimitive::Offset {
        dx: distance,
        dy: -distance,
    };
    let shadow = |distance: f32, std_deviation: f32, keeps_source: bool| {
        let color = AlphaColor::from_rgba8(255, 0, 0, 255);
        let (dx, dy, edge_mode) = (distance, -distance, EdgeMode::None);
        if keeps_source {
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode,
            }
        } else {
            FilterPrimitive::DropShadowOnly {
                dx,
                dy,
                std_deviation,
                color,
                edge_mode,
            }
        }
    };
    let blur = FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    };
    // At larger scales, every distance exceeds `f32` in the target's space.
    for scale in [1.0, 4.0, 1.0e30] {
        for far in [3.2e9, 1.0e19, f32::MAX] {
            for std_deviation in [0.0, 2.0] {
                // Each nesting, and the same without the moves that cancel.
                let cases = [
                    (vec![offset(far), offset(-far)], vec![]),
                    (
                        vec![offset(-far), offset(4.0), offset(far)],
                        vec![offset(4.0)],
                    ),
                    (
                        vec![offset(far), blur.clone(), offset(-far)],
                        vec![blur.clone()],
                    ),
                    (
                        vec![offset(far), shadow(-far, std_deviation, true)],
                        vec![shadow(0.0, std_deviation, false)],
                    ),
                    (
                        vec![shadow(far, std_deviation, true), offset(-far)],
                        vec![shadow(0.0, std_deviation, false)],
                    ),
                    (
                        vec![offset(-far), shadow(far, std_deviation, false), offset(4.0)],
                        vec![shadow(4.0, std_deviation, false)],
                    ),
                ];
                // The outer shadow reads the inner one's shadow, not its source.
                let shadow_of_shadow = (
                    vec![
                        shadow(far, std_deviation, false),
                        shadow(-far, std_deviation, true),
                    ],
                    vec![
                        shadow(0.0, std_deviation, false),
                        shadow(0.0, std_deviation, false),
                    ],
                );
                // The outer blur reads the inner one's output at full
                // resolution throughout its reach, which takes work growing
                // with the square of both deviations.
                let cases = cases
                    .into_iter()
                    .chain((f64::from(std_deviation) * scale <= 8.0).then_some(shadow_of_shadow));
                for (filters, reference) in cases {
                    let expected = nested_in(&reference, scale);
                    assert!(
                        expected.iter().any(|&value| value != 0) || scale > 4.0,
                        "{reference:?}"
                    );
                    assert!(
                        nested_in(&filters, scale) == expected,
                        "{filters:?} at scale {scale}"
                    );
                }
            }
        }
    }
}

/// Blur, in a 64 by 64 view, squares lying `distance` pixels away from it on
/// either side with a narrow blur nested in a blur by `std_deviation`.
fn far_squares_in_nested_blurs(std_deviation: f32, distance: f64) -> Vec<u8> {
    let blur = |std_deviation| {
        Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation,
            edge_mode: EdgeMode::None,
        })
    };
    let mut context = context(64, 64);
    context.push_filter_layer(blur(std_deviation));
    context.push_filter_layer(blur(100.0));
    context.set_paint(BLUE);
    for corner in [-distance, distance] {
        context.fill_rect(&Rect::new(corner, corner, corner + 64.0, corner + 64.0));
    }
    context.pop_layer();
    context.pop_layer();
    render(context)
}

#[test]
fn blur_nested_in_a_wider_one_holds_the_same_memory_however_wide_that_is() {
    // The outer blur reads the inner one where it has content, around each
    // square, not across the whole reach between them.
    let (narrow, _) = measure(|| far_squares_in_nested_blurs(10_000.0, 25_000.0));
    for std_deviation in [20_000.0, 1.0e6, 1.0e10] {
        let (usage, _) = measure(|| far_squares_in_nested_blurs(std_deviation, 25_000.0));
        assert!(
            usage.peak <= narrow.peak + narrow.peak / 4,
            "outer {std_deviation}: {usage:?}, at 10000: {narrow:?}"
        );
    }
}

#[test]
fn offset_past_f32_in_a_scaled_space_is_not_undone_by_a_smaller_one() {
    // The inner offset moves the square by twice `f32::MAX` in the target's
    // space, and the outer one back by `f32::MAX`, which leaves it out of view.
    let offset = |distance: f32| FilterPrimitive::Offset {
        dx: distance,
        dy: distance,
    };
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(offset(-f32::MAX)));
    context.set_transform(Affine::scale(2.0));
    context.push_filter_layer(Filter::from_primitive(offset(f32::MAX)));
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(8.0, 8.0, 24.0, 24.0));
    context.pop_layer();
    context.pop_layer();
    assert!(render(context).iter().all(|&value| value == 0));
}

/// Render, in a 64 by 64 view under an offset by `-far`, a red shadow of a
/// square and a smaller green square, each moved `far` by a nested filter and
/// composited with each other there, with a square `drawn` in the view itself.
fn far_copies(far: f32, std_deviation: f32, drawn: bool) -> Vec<u8> {
    let mut context = context(64, 64);
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: -far,
        dy: far,
    }));
    if drawn {
        context.set_paint(BLUE);
        context.fill_rect(&Rect::new(0.0, 0.0, 64.0, 64.0));
    }
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadowOnly {
        dx: far,
        dy: -far,
        std_deviation,
        color: AlphaColor::from_rgba8(255, 0, 0, 255),
        edge_mode: EdgeMode::None,
    }));
    context.set_paint(BLUE);
    context.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
    context.pop_layer();
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: far,
        dy: -far,
    }));
    context.set_paint(AlphaColor::from_rgba8(0, 255, 0, 255));
    context.fill_rect(&Rect::new(24.0, 24.0, 40.0, 40.0));
    context.pop_layer();
    context.pop_layer();
    render(context)
}

#[test]
fn far_copies_composite_with_each_other_before_moving_back() {
    for std_deviation in [0.0, 2.0] {
        let expected = far_copies(0.0, std_deviation, false);
        assert!(
            expected
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[1] == 255)
        );
        for far in [3.2e9, 1.0e19, f32::MAX] {
            assert!(
                far_copies(far, std_deviation, true) == expected,
                "{far}, {std_deviation}"
            );
        }
    }
}

/// The error function, within 1.5e-7 (Abramowitz and Stegun, 7.1.26).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    1.0 - poly * (-x * x).exp()
}
