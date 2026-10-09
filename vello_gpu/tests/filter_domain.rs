// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mathematical filter-pass oracles with nonzero atlas origins and odd extents.
#![cfg(feature = "wgpu")]

use vello_gpu::{
    ClearSettings, RenderSize, RenderTargetConfig, Renderer, Resources, Scene, TargetInit,
    TextureBindings,
};
use wgpu::util::DeviceExt;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("filter regression requires a wgpu adapter; software Vulkan is sufficient");
        eprintln!("filter domain adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("create filter regression device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production filter shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::FILTER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("filter domain regression"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 36,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0=>Uint32,1=>Uint32,2=>Uint32,3=>Uint32,4=>Uint32,5=>Uint32,6=>Uint32,7=>Uint32,8=>Uint32],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
        }
    }

    fn pass(
        &self,
        pixels: &[[u8; 4]],
        size: [u32; 2],
        dest: [u32; 2],
        kind: u32,
        parameters: [u32; 12],
    ) -> Vec<[u8; 4]> {
        const ATLAS: u32 = 32;
        let extent = wgpu::Extent3d {
            width: ATLAS,
            height: ATLAS,
            depth_or_array_layers: 1,
        };
        let source = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("filter source"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let source_origin = [7, 9];
        let mut atlas = vec![[0_u8; 4]; (ATLAS * ATLAS) as usize];
        for y in 0..size[1] {
            for x in 0..size[0] {
                atlas[((y + source_origin[1]) * ATLAS + x + source_origin[0]) as usize] =
                    pixels[(y * size[0] + x) as usize];
            }
        }
        self.queue.write_texture(
            source.as_image_copy(),
            bytemuck::cast_slice(&atlas),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS * 4),
                rows_per_image: None,
            },
            extent,
        );
        let data = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("filter parameters"),
            size: wgpu::Extent3d {
                width: 3,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            data.as_image_copy(),
            bytemuck::cast_slice(&parameters),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(48),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 3,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("filter result"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let source_view = source.create_view(&wgpu::TextureViewDescriptor::default());
        let data_view = data.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let groups = [
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&data_view),
                }],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                }],
            }),
        ];
        let pack = |xy: [u32; 2]| xy[0] | (xy[1] << 16);
        let dest_origin = [3, 4];
        let fields = [
            pack(source_origin),
            pack(size),
            pack(dest_origin),
            pack(dest),
            pack([ATLAS, ATLAS]),
            0,
            pack(source_origin),
            pack(dest),
            kind,
        ];
        let instances = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&fields),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * u64::from(ATLAS),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            for (index, group) in groups.iter().enumerate() {
                pass.set_bind_group(u32::try_from(index).unwrap(), group, &[]);
            }
            pass.set_vertex_buffer(0, instances.slice(..));
            pass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            extent,
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("map filter output");
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("complete filter pass");
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let mut result = Vec::new();
        for y in 0..dest[1] {
            for x in 0..dest[0] {
                let offset = ((y + dest_origin[1]) * 256 + (x + dest_origin[0]) * 4) as usize;
                result.push(mapped[offset..offset + 4].try_into().unwrap());
            }
        }
        result
    }
}

fn extend(value: i32, size: i32, mode: u32) -> Option<i32> {
    match mode {
        0 => Some(value.clamp(0, size - 1)),
        1 => Some(value.rem_euclid(size)),
        2 => {
            let folded = value.rem_euclid(size * 2);
            Some(if folded < size {
                folded
            } else {
                size * 2 - 1 - folded
            })
        }
        3 => (0..size).contains(&value).then_some(value),
        _ => unreachable!(),
    }
}

fn sample(input: &[[u8; 4]], size: [u32; 2], x: i32, y: i32, mode: u32) -> [f64; 4] {
    let (Some(x), Some(y)) = (
        extend(x, size[0] as i32, mode),
        extend(y, size[1] as i32, mode),
    ) else {
        return [0.0; 4];
    };
    input[(y * size[0] as i32 + x) as usize].map(f64::from)
}

#[test]
fn edge_modes_cover_blur_and_odd_pyramid_passes() {
    let gpu = Gpu::new();
    for size in [[1_u32, 1], [3, 5], [5, 3]] {
        let input: Vec<_> = (0..size[0] * size[1])
            .map(|i| [((i * 11 + 16) % 192) as u8, 0, 0, 255])
            .collect();
        for mode in 0..4 {
            let mut parameters = [0; 12];
            // An exactly representable three-tap kernel [1,2,1]/4.
            parameters[0] = 2 | (mode << 5) | (1 << 11);
            parameters[1] = 0.5_f32.to_bits();
            parameters[2] = 0.25_f32.to_bits();
            parameters[5] = 1.0_f32.to_bits();
            for kind in [3, 4, 5, 6] {
                let dest = match kind {
                    3 => [size[0].div_ceil(2), size[1].div_ceil(2)],
                    6 => [size[0] * 2 - 1, size[1] * 2 - 1],
                    _ => size,
                };
                let actual = gpu.pass(&input, size, dest, kind, parameters);
                for y in 0..dest[1] {
                    for x in 0..dest[0] {
                        let x = x as i32;
                        let y = y as i32;
                        let taps: Vec<_> = match kind {
                            3 => (-1..=2)
                                .flat_map(|dy| {
                                    (-1..=2).map(move |dx| {
                                        (
                                            x * 2 + dx,
                                            y * 2 + dy,
                                            f64::from(
                                                [1, 3, 3, 1][(dx + 1) as usize]
                                                    * [1, 3, 3, 1][(dy + 1) as usize],
                                            ) / 64.0,
                                        )
                                    })
                                })
                                .collect(),
                            4 => vec![(x - 1, y, 0.25), (x, y, 0.5), (x + 1, y, 0.25)],
                            5 => vec![(x, y - 1, 0.25), (x, y, 0.5), (x, y + 1, 0.25)],
                            6 => {
                                let sx = if x % 2 == 0 { -1 } else { 1 };
                                let sy = if y % 2 == 0 { -1 } else { 1 };
                                vec![
                                    (x / 2, y / 2, 0.5625),
                                    (x / 2 + sx, y / 2, 0.1875),
                                    (x / 2, y / 2 + sy, 0.1875),
                                    (x / 2 + sx, y / 2 + sy, 0.0625),
                                ]
                            }
                            _ => unreachable!(),
                        };
                        let mut expected = [0.0; 4];
                        for (sx, sy, weight) in taps {
                            let pixel = sample(&input, size, sx, sy, mode);
                            for channel in 0..4 {
                                expected[channel] += pixel[channel] * weight;
                            }
                        }
                        let actual = actual[(y as u32 * dest[0] + x as u32) as usize];
                        for channel in 0..4 {
                            // Rgba8Unorm conversion can choose either adjacent byte at an exact half.
                            assert!(
                                (f64::from(actual[channel]) - expected[channel]).abs() <= 0.501,
                                "size={size:?} mode={mode} kind={kind} pixel=({x},{y}) channel={channel}: actual={actual:?}, exact={expected:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn offsets_round_negative_half_away_from_zero() {
    let gpu = Gpu::new();
    let size = [5, 3];
    let input: Vec<_> = (0..15).map(|i| [i * 8, 0, 0, 255]).collect();
    for (offset, pixels) in [(-1.5_f32, -2), (-0.5, -1), (0.5, 1), (1.5, 2)] {
        for shadow in [false, true] {
            let mut parameters = [0; 12];
            if shadow {
                parameters[0] = 3;
                parameters[8] = offset.to_bits();
                parameters[9] = offset.to_bits();
            } else {
                parameters[1] = offset.to_bits();
                parameters[2] = offset.to_bits();
            }
            let actual = gpu.pass(&input, size, size, 2, parameters);
            for y in 0..3 {
                for x in 0..5 {
                    let expected = sample(&input, size, x - pixels, y - pixels, 3);
                    assert_eq!(
                        actual[(y * 5 + x) as usize].map(f64::from),
                        expected,
                        "offset={offset}, shadow={shadow}, pixel=({x},{y})"
                    );
                }
            }
        }
    }
}

fn render_pixels(
    renderer: &mut Renderer,
    resources: &mut Resources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &Scene,
    use_depth: bool,
) -> Vec<u8> {
    let size = RenderSize {
        width: scene.width(),
        height: scene.height(),
    };
    let extent = wgpu::Extent3d {
        width: size.width.into(),
        height: size.height.into(),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("intermediate bounds output"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = use_depth.then(|| Renderer::create_depth_texture_view(device, &size));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    renderer
        .render(
            scene,
            resources,
            device,
            queue,
            &mut encoder,
            &size,
            &view,
            depth.as_ref(),
            &TextureBindings::new(),
            TargetInit::Clear(ClearSettings::default()),
        )
        .expect("render odd-sized intermediate scene");
    let row_bytes = (extent.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("intermediate bounds readback"),
        size: u64::from(row_bytes) * u64::from(extent.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: None,
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map intermediate bounds output");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("complete intermediate bounds rendering");
    let mut pixels = Vec::new();
    {
        let data = buffer.slice(..).get_mapped_range().unwrap();
        for row in data.chunks_exact(row_bytes as usize) {
            pixels.extend_from_slice(&row[..usize::from(size.width) * 4]);
        }
    }
    buffer.unmap();
    pixels
}

#[test]
fn nested_offset_layers_round_once_per_layer() {
    use vello_common::color::palette::css::RED;
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::kurbo::Rect;
    let gpu = Gpu::new();
    let config = RenderTargetConfig {
        width: 13,
        height: 9,
        format: wgpu::TextureFormat::Rgba8Unorm,
    };
    let (mut renderer, mut resources) = Renderer::new(&gpu.device, &config);
    for shadow in [false, true] {
        for nested in [false, true] {
            let mut scene = Scene::new(13, 9);
            let effect = if shadow {
                FilterPrimitive::DropShadowOnly {
                    dx: -0.5,
                    dy: -1.5,
                    std_deviation: 0.0,
                    color: RED,
                    edge_mode: EdgeMode::None,
                }
            } else {
                FilterPrimitive::Offset { dx: -0.5, dy: -1.5 }
            };
            scene.push_filter_layer(Filter::from_primitive(effect.clone()));
            if nested {
                scene.push_filter_layer(Filter::from_primitive(effect));
            }
            scene.set_paint(RED);
            scene.fill_rect(&Rect::new(5.0, 5.0, 8.0, 8.0));
            if nested {
                scene.pop_layer();
            }
            scene.pop_layer();
            let pixels = render_pixels(
                &mut renderer,
                &mut resources,
                &gpu.device,
                &gpu.queue,
                &scene,
                false,
            );
            let shift = if nested { 2 } else { 1 };
            for (index, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
                let x = index % 13;
                let y = index / 13;
                let expected = if (5 - shift..8 - shift).contains(&x)
                    && (5 - 2 * shift..8 - 2 * shift).contains(&y)
                {
                    [255, 0, 0, 255]
                } else {
                    [0; 4]
                };
                assert_eq!(
                    *pixel, expected,
                    "shadow={shadow}, nested={nested}, pixel=({x},{y})"
                );
            }
        }
    }
}

#[test]
fn paired_blur_taps_extend_each_discrete_endpoint() {
    let gpu = Gpu::new();
    let input = [[32, 0, 0, 255], [128, 0, 0, 255], [224, 0, 0, 255]];
    // [1,3,4,16,4,3,1]/32: the first positive pair straddles two texels,
    // while the second reaches a full image width beyond an edge.
    let weights = [1.0, 3.0, 4.0, 16.0, 4.0, 3.0, 1.0];
    for mode in 0..4 {
        let mut parameters = [0; 12];
        parameters[0] = 2 | (mode << 5) | (2 << 11);
        parameters[1] = 0.5_f32.to_bits();
        parameters[2] = (7.0_f32 / 32.0).to_bits();
        parameters[3] = (1.0_f32 / 32.0).to_bits();
        parameters[5] = (10.0_f32 / 7.0).to_bits();
        parameters[6] = 3.0_f32.to_bits();
        for (kind, size) in [(4, [3, 1]), (5, [1, 3])] {
            let actual = gpu.pass(&input, size, size, kind, parameters);
            for (index, pixel) in actual.iter().enumerate() {
                let mut expected = [0.0; 4];
                for (tap, weight) in weights.iter().enumerate() {
                    let coord = i32::try_from(index).unwrap() + i32::try_from(tap).unwrap() - 3;
                    let (x, y) = if kind == 4 { (coord, 0) } else { (0, coord) };
                    let sample = sample(&input, size, x, y, mode);
                    for channel in 0..4 {
                        expected[channel] += sample[channel] * weight / 32.0;
                    }
                }
                for channel in 0..4 {
                    assert!(
                        (f64::from(pixel[channel]) - expected[channel]).abs() <= 0.501,
                        "mode={mode}, kind={kind}, index={index}: actual={pixel:?}, exact={expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn empty_destructive_filters_preserve_empty_source() {
    use vello_common::color::palette::css::RED;
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::kurbo::{Rect, Shape};
    use vello_common::peniko::{BlendMode, Compose, Mix};

    let gpu = Gpu::new();
    let config = RenderTargetConfig {
        width: 13,
        height: 9,
        format: wgpu::TextureFormat::Rgba8Unorm,
    };
    let (mut renderer, mut resources) = Renderer::new(&gpu.device, &config);
    let filters = [
        FilterPrimitive::GaussianBlur {
            std_deviation: 100_000.0,
            edge_mode: EdgeMode::None,
        },
        FilterPrimitive::GaussianBlur {
            std_deviation: 1.0,
            edge_mode: EdgeMode::Duplicate,
        },
        FilterPrimitive::DropShadow {
            dx: -100_000.0,
            dy: 100_000.0,
            std_deviation: 100_000.0,
            color: RED,
            edge_mode: EdgeMode::Wrap,
        },
        FilterPrimitive::Offset {
            dx: -100_000.0,
            dy: 100_000.0,
        },
        // Flood still has an empty filter region when there is no source geometry.
        FilterPrimitive::Flood { color: RED },
    ];
    for (index, primitive) in filters.into_iter().enumerate() {
        for compose in [Compose::Clear, Compose::Copy] {
            for clipped in [false, true] {
                let mut scene = Scene::new(13, 9);
                scene.set_paint(RED);
                scene.fill_rect(&Rect::new(0.0, 0.0, 13.0, 9.0));
                let clip = Rect::new(2.0, 3.0, 9.0, 7.0).to_path(0.1);
                scene.push_layer(
                    clipped.then_some(&clip),
                    Some(BlendMode::new(Mix::Normal, compose)),
                    None,
                    None,
                    Some(Filter::from_primitive(primitive.clone())),
                );
                scene.pop_layer();
                let pixels = render_pixels(
                    &mut renderer,
                    &mut resources,
                    &gpu.device,
                    &gpu.queue,
                    &scene,
                    false,
                );
                for (pixel_index, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
                    let x = pixel_index % 13;
                    let y = pixel_index / 13;
                    let expected = if !clipped || (2..9).contains(&x) && (3..7).contains(&y) {
                        [0; 4]
                    } else {
                        [255, 0, 0, 255]
                    };
                    assert_eq!(
                        *pixel, expected,
                        "filter={index}, compose={compose:?}, clipped={clipped}, pixel=({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn tiny_filter_transform_matches_its_effective_blur() {
    use vello_common::color::palette::css::RED;
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::kurbo::{Affine, Rect};
    let gpu = Gpu::new();
    let config = RenderTargetConfig {
        width: 16,
        height: 16,
        format: wgpu::TextureFormat::Rgba8Unorm,
    };
    let (mut renderer, mut resources) = Renderer::new(&gpu.device, &config);
    let mut frames = Vec::new();
    for (scale, std_deviation) in [(1.0, 0.1), (1e-12, 1e11)] {
        let mut scene = Scene::new(16, 16);
        scene.set_transform(Affine::scale(scale));
        scene.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation,
            edge_mode: EdgeMode::None,
        }));
        scene.reset_transform();
        scene.set_paint(RED);
        scene.fill_rect(&Rect::new(4.0, 4.0, 8.0, 8.0));
        scene.pop_layer();
        frames.push(render_pixels(
            &mut renderer,
            &mut resources,
            &gpu.device,
            &gpu.queue,
            &scene,
            false,
        ));
    }
    assert_eq!(
        frames[0], frames[1],
        "preparation and source padding must use the same effective transform scale"
    );
}
