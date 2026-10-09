// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur filter passes.

import package::filters::sample::{get_edge_mode, sample_linear};

const BLUR_HORIZONTAL: vec2<f32> = vec2<f32>(1.0, 0.0);
const BLUR_VERTICAL: vec2<f32> = vec2<f32>(0.0, 1.0);

/// Number of linear taps stored in the packed filter header.
fn get_n_linear_taps(texel0: vec4<u32>) -> u32 { return (texel0.x >> 11u) & 0x3u; }

/// Center weight for gaussian blur convolution.
fn get_center_weight(texel0: vec4<u32>) -> f32 { return bitcast<f32>(texel0.y); }

/// Linear sample weights for gaussian blur convolution.
fn get_linear_weights(texel0: vec4<u32>, texel1: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel0.z),
        bitcast<f32>(texel0.w),
        bitcast<f32>(texel1.x),
    );
}

/// Linear sample offsets for gaussian blur convolution.
fn get_linear_offsets(texel1: vec4<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(texel1.y),
        bitcast<f32>(texel1.z),
        bitcast<f32>(texel1.w),
    );
}

fn gaussian_blur(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    source_origin: vec2<u32>,
    source_size: vec2<u32>,
    source_rel: vec2<f32>,
    direction: vec2<f32>,
    texel0: vec4<u32>,
    texel1: vec4<u32>,
) -> vec4<f32> {
    let edge_mode = get_edge_mode(texel0);
    let n_linear_taps = get_n_linear_taps(texel0);
    let center_weight = get_center_weight(texel0);

    var color = sample_linear(source_texture, linear_sampler, source_origin,
        source_size, source_rel, vec2(0.0), edge_mode) * center_weight;

    let linear_weights = get_linear_weights(texel0, texel1);
    let linear_offsets = get_linear_offsets(texel1);

    if n_linear_taps > 0u {
        let distance = direction * linear_offsets.x;
        color += sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, distance, edge_mode) * linear_weights.x;
        color += sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, -distance, edge_mode) * linear_weights.x;
    }
    if n_linear_taps > 1u {
        let distance = direction * linear_offsets.y;
        color += sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, distance, edge_mode) * linear_weights.y;
        color += sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, -distance, edge_mode) * linear_weights.y;
    }
    if n_linear_taps > 2u {
        let distance = direction * linear_offsets.z;
        color += sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, distance, edge_mode) * linear_weights.z;
        color += sample_linear(source_texture, linear_sampler, source_origin, source_size, source_rel, -distance, edge_mode) * linear_weights.z;
    }

    return color;
}
