// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter scaling passes.

import package::filters::sample::sample_linear;

// Downscale with a separable [1,3,3,1]/8 filter. Native bilinear sampling combines
// the horizontal and vertical passes and reduces the sample count from 16 to four.
fn downscale(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    position: vec4<f32>,
    source_origin: vec2<u32>,
    source_size: vec2<u32>,
    edge_mode: u32,
    dest_origin: vec2<u32>,
) -> vec4<f32> {
    let frag_coord = vec2<u32>(position.xy);
    let rel = vec2<i32>(frag_coord - dest_origin);
    let source_rel = vec2<f32>(rel * 2);

    // Shift the two sample points to produce weights [1,3,3,1].
    let lo = vec2<f32>(-0.25);
    let hi = vec2<f32>(1.25);

    let s00 = sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, vec2(lo.x, lo.y), edge_mode);
    let s01 = sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, vec2(lo.x, hi.y), edge_mode);
    let s10 = sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, vec2(hi.x, lo.y), edge_mode);
    let s11 = sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, vec2(hi.x, hi.y), edge_mode);

    return (s00 + s01 + s10 + s11) * 0.25;
}

fn upscale(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    position: vec4<f32>,
    source_origin: vec2<u32>,
    source_size: vec2<u32>,
    edge_mode: u32,
    dest_origin: vec2<u32>,
) -> vec4<f32> {
    let frag_coord = vec2<u32>(position.xy);
    let rel = vec2<i32>(frag_coord - dest_origin);
    let source_base = vec2<f32>(rel / 2);
    let phase = vec2<f32>(rel % 2);

    // For even phases: 75% of current, 25% of top/left.
    // For odd phases: 75% of current, 25% of bottom/right.
    let sample_offset = select(vec2(-0.25), vec2(0.25), phase == vec2(1.0));

    return sample_linear(source_texture, linear_sampler, source_origin, source_size, source_base, sample_offset, edge_mode);
}
