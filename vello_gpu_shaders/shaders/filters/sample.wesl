// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sample within one filter region without crossing into neighboring atlas allocations.

fn get_edge_mode(texel0: vec4<u32>) -> u32 { return (texel0.x >> 5u) & 3u; }

fn extend_coordinate(coord: vec2<i32>, size: vec2<i32>, edge_mode: u32) -> vec2<i32> {
    if edge_mode == 0u {
        return clamp(coord, vec2(0), size - 1);
    }
    let period = select(size * 2, size, edge_mode == 1u);
    let wrapped = ((coord % period) + period) % period;
    return select(period - wrapped - 1, wrapped, wrapped < size);
}

fn sample_linear(
    source_texture: texture_2d<f32>,
    linear_sampler: sampler,
    source_origin: vec2<u32>,
    source_size: vec2<u32>,
    base: vec2<f32>,
    offset: vec2<f32>,
    edge_mode: u32,
) -> vec4<f32> {
    let rel_coord = base + offset;
    // Transparent atlas padding already implements None. Interior taps of other
    // modes keep the same hardware bilinear path and its arithmetic.
    if edge_mode == 3u || (all(rel_coord >= vec2(0.0)) && all(rel_coord <= vec2<f32>(source_size) - 1.0)) {
        let texel = vec2<f32>(source_origin) + base + offset;
        return textureSampleLevel(source_texture, linear_sampler,
            (texel + 0.5) / vec2<f32>(textureDimensions(source_texture)), 0.0);
    }

    // Mapping each discrete endpoint preserves wrap seams and the repeated edge
    // texel in mirror mode. Mapping a fractional coordinate would blend against
    // unrelated atlas padding at those seams.
    let first = vec2<i32>(floor(rel_coord));
    let fraction = fract(rel_coord);
    let origin = vec2<i32>(source_origin);
    let size = vec2<i32>(source_size);
    let p00 = textureLoad(source_texture, origin + extend_coordinate(first, size, edge_mode), 0);
    let p10 = textureLoad(source_texture, origin + extend_coordinate(first + vec2(1, 0), size, edge_mode), 0);
    let p01 = textureLoad(source_texture, origin + extend_coordinate(first + vec2(0, 1), size, edge_mode), 0);
    let p11 = textureLoad(source_texture, origin + extend_coordinate(first + vec2(1, 1), size, edge_mode), 0);
    return mix(mix(p00, p10, fraction.x), mix(p01, p11, fraction.x), fraction.y);
}
