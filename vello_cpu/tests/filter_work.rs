// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory a filter layer allocates follows the visible view, not how far
//! its source lies, how wide its blur is or how deeply filters are nested.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use vello_cpu::color::AlphaColor;
use vello_cpu::color::palette::css::BLUE;
use vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_cpu::kurbo::{Affine, Rect};
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
    assert!(expected.chunks_exact(4).any(|pixel| pixel[3] != 0));
    for distance in [4_096.0, 60_000.0, 262_144.0, 16_777_216.0] {
        let (usage, pixels) = measure(|| moved_into_view(distance));
        assert!(pixels == expected, "pixels at distance {distance}");
        assert!(
            usage.peak <= near.peak + near.peak / 4 && usage.total <= near.total + near.total / 4,
            "distance {distance}: {usage:?}, at 96: {near:?}"
        );
    }
}

/// Render a blur of standard deviation `std_deviation` over a square that
/// covers all it reaches of a 12 by 12 view.
/// Blur a square that covers the view and reaches four standard deviations
/// past it, drawn and blurred in a space scaled by `scale`.
fn blurred_square(std_deviation: f32, scale: f64) -> Vec<u8> {
    let mut context = context(12, 12);
    context.set_transform(Affine::scale(scale));
    context.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    }));
    context.set_paint(BLUE);
    let (reach, size) = (4.0 * f64::from(std_deviation), 12.0 / scale);
    context.fill_rect(&Rect::new(-reach, -reach, size + reach, size + reach));
    context.pop_layer();
    render(context)
}

#[test]
fn blur_allocates_a_bounded_amount_however_wide_it_is() {
    // At larger scales, the largest standard deviations exceed `f32` in the
    // target's space.
    for scale in [1.0, 4.0, 1.0e30] {
        for std_deviation in [4.0, 40.0, 400.0, 4_000.0, 20_000.0, 1.0e9, 1.0e30, f32::MAX] {
            let (usage, pixels) = measure(|| blurred_square(std_deviation, scale));
            // The blur reads three standard deviations around each pixel, which
            // hold 99.7% of its weight.
            for pixel in pixels.chunks_exact(4) {
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
