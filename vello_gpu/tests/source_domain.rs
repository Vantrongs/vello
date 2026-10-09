// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wide source paint coordinates remain independent from physical texture limits.
#![cfg(feature = "wgpu")]

use vello_common::color::palette::css::{BLUE, RED};
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Vec2};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{BlendMode, Compose, Gradient, ImageSampler, Mix};
use vello_common::pixmap::{PixelMetadata, Pixmap};
use vello_gpu::{
    ClearSettings, Config, GpuStrip, IntermediateTextureError, RenderError, RenderSize,
    RenderTargetConfig, Renderer, Resources, Scene, TargetInit, TextureBindings,
};
use wgpu::util::DeviceExt;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("source-domain regression requires a wgpu adapter; software Vulkan is sufficient");
        eprintln!("source domain adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("create source-domain regression device");
        Self { device, queue }
    }

    fn texture(
        &self,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("source domain texture"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn upload(&self, texture: &wgpu::Texture, bytes: &[u8], row_bytes: u32) {
        self.queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: None,
            },
            texture.size(),
        );
    }

    fn readback(&self, mut encoder: wgpu::CommandEncoder, texture: &wgpu::Texture) -> Vec<u8> {
        let row_bytes = (texture.width() * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("source domain readback"),
            size: u64::from(row_bytes) * u64::from(texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("map source domain output");
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("complete source domain rendering");
        let mut pixels = Vec::new();
        {
            let data = buffer.slice(..).get_mapped_range().unwrap();
            for row in data.chunks_exact(row_bytes as usize) {
                pixels.extend_from_slice(&row[..texture.width() as usize * 4]);
            }
        }
        buffer.unmap();
        pixels
    }

    fn scene(
        &self,
        renderer: &mut Renderer,
        resources: &mut Resources,
        scene: &Scene,
    ) -> Result<Vec<u8>, RenderError> {
        let size = RenderSize {
            width: scene.width(),
            height: scene.height(),
        };
        let texture = self.texture(
            [size.width.into(), size.height.into()],
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        renderer.render(
            scene,
            resources,
            &self.device,
            &self.queue,
            &mut encoder,
            &size,
            &view,
            None,
            &TextureBindings::new(),
            TargetInit::Clear(ClearSettings::default()),
        )?;
        Ok(self.readback(encoder, &texture))
    }

    // Feed independent paint data through the production render shader and its public vertex layout.
    // The scheduler unit tests separately verify source-to-atlas geometry rebasing and payload creation.
    fn paint(&self, size: [u16; 2], origin: [u32; 2], gradient: bool) -> Vec<u8> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("production source paint shader"),
                source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::RENDER.into()),
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("source paint coordinates"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<GpuStrip>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &GpuStrip::vertex_attributes(),
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let config = Config {
            width: size[0].into(),
            height: size[1].into(),
            strip_height: 4,
            alphas_tex_width_bits: 0,
            encoded_paints_tex_width_bits: 2,
            strip_offset_x: 0,
            strip_offset_y: 0,
            negate_ndc: 0,
        };
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("source paint config"),
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let local = origin.iter().any(|&value| value > u32::from(u16::MAX));
        let strip = GpuStrip {
            x: 0,
            y: 0,
            width: size[0],
            dense_width_or_rect_height: size[1],
            col_idx_or_rect_frac: 0,
            payload: if local { 0.0_f32.to_bits() } else { origin[0] },
            payload_y: if local { 0.0_f32.to_bits() } else { origin[1] },
            paint_and_rect_flag: (1 << 31)
                | ((if gradient { 2 } else { 1 }) << 26)
                | (u32::from(local) << 25),
            depth_index: 0,
        };
        let vertices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wide source strip"),
                contents: bytemuck::bytes_of(&strip),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let sampled = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let alpha = self.texture([1, 1], wgpu::TextureFormat::Rgba32Uint, sampled);
        let image = self.texture(
            size.map(u32::from),
            wgpu::TextureFormat::Rgba8Unorm,
            sampled,
        );
        let image_pixels: Vec<[u8; 4]> = (0..size[1])
            .flat_map(|y| {
                (0..size[0]).map(move |x| {
                    [
                        u8::try_from(x * 13).unwrap(),
                        u8::try_from(y * 13).unwrap(),
                        71,
                        255,
                    ]
                })
            })
            .collect();
        self.upload(
            &image,
            bytemuck::cast_slice(&image_pixels),
            u32::from(size[0]) * 4,
        );
        let lut = self.texture([16, 1], wgpu::TextureFormat::Rgba8Unorm, sampled);
        let lut_pixels: Vec<[u8; 4]> = (0_u8..16)
            .map(|v| [v * 13, 47, 255 - v * 13, 255])
            .collect();
        self.upload(&lut, bytemuck::cast_slice(&lut_pixels), 64);
        let paints = self.texture([4, 1], wgpu::TextureFormat::Rgba32Uint, sampled);
        let mut encoded = [0_u32; 16];
        if gradient {
            let transpose = size[1] > size[0];
            let scale = 1.0 / f32::from(if transpose { size[1] } else { size[0] });
            encoded[0] = 16;
            encoded[if transpose { 4 } else { 2 }] = scale.to_bits();
            encoded[6] = (-(origin[usize::from(transpose)] as f32) * scale).to_bits();
        } else {
            encoded[1] = (u32::from(size[0]) << 16) | u32::from(size[1]);
            encoded[3] = 1.0_f32.to_bits();
            encoded[6] = 1.0_f32.to_bits();
            encoded[7] = (-(origin[0] as f32)).to_bits();
            encoded[8] = (-(origin[1] as f32)).to_bits();
            encoded[9] = u32::MAX;
            encoded[10] = 1;
        }
        self.upload(&paints, bytemuck::cast_slice(&encoded), 64);
        let alpha_view = alpha.create_view(&wgpu::TextureViewDescriptor::default());
        let image_view = image.create_view(&wgpu::TextureViewDescriptor::default());
        let lut_view = lut.create_view(&wgpu::TextureViewDescriptor::default());
        let paints_view = paints.create_view(&wgpu::TextureViewDescriptor::default());
        let groups = [
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&alpha_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&image_view),
                    },
                ],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                }],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&paints_view),
                }],
            }),
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(3),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                }],
            }),
        ];
        let output = self.texture(
            size.map(u32::from),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wide source paints"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            for (index, group) in groups.iter().enumerate() {
                pass.set_bind_group(u32::try_from(index).unwrap(), group, &[]);
            }
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.draw(0..4, 0..1);
        }
        self.readback(encoder, &output)
    }
}

#[test]
fn wide_image_and_gradient_coordinates_match_local_render() {
    let gpu = Gpu::new();
    for size in [[16, 4], [4, 16]] {
        for gradient in [false, true] {
            let reference = gpu.paint(size, [0, 0], gradient);
            for (index, actual) in reference.as_chunks::<4>().0.iter().enumerate() {
                let x = u16::try_from(index % usize::from(size[0])).unwrap();
                let y = u16::try_from(index / usize::from(size[0])).unwrap();
                let expected = if gradient {
                    let axis = if size[1] > size[0] { y } else { x };
                    // floor(((axis + 0.5) / 16) * (16 - 1)) using integer arithmetic.
                    let value = u8::try_from((2 * axis + 1) * 15 / 32).unwrap();
                    [value * 13, 47, 255 - value * 13, 255]
                } else {
                    [
                        u8::try_from(x * 13).unwrap(),
                        u8::try_from(y * 13).unwrap(),
                        71,
                        255,
                    ]
                };
                assert_eq!(
                    *actual, expected,
                    "size={size:?}, gradient={gradient}, pixel=({x},{y})"
                );
            }
            for origin in [
                [70000, 0],
                [0, 70000],
                [262148, 131070],
                [196607, 262148],
                [16_777_221, 16_777_222],
            ] {
                assert_eq!(
                    gpu.paint(size, origin, gradient),
                    reference,
                    "size={size:?} origin={origin:?} gradient={gradient}"
                );
            }
        }
    }
}

#[test]
fn large_offsets_keep_small_gradient_and_image_sources_tight() {
    let gpu = Gpu::new();
    let mut mismatches = Vec::new();
    for (width, height) in [(16, 4), (4, 16)] {
        let (mut renderer, mut resources) = Renderer::new(
            &gpu.device,
            &RenderTargetConfig {
                width,
                height,
                format: wgpu::TextureFormat::Rgba8Unorm,
            },
        );
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        u8::try_from(x * 13).unwrap(),
                        u8::try_from(y * 13).unwrap(),
                        71,
                        255,
                    ]
                })
            })
            .collect();
        let pixmap = Pixmap::from_parts(pixels, width, height, PixelMetadata::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let id = renderer.upload_image(
            &mut resources,
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &pixmap,
        );
        gpu.queue.submit([encoder.finish()]);
        let image = Image {
            image: ImageSource::opaque_id(id),
            sampler: ImageSampler::default(),
        };
        let gradient = Gradient::new_linear((0.0, 0.0), (f64::from(width), f64::from(height)))
            .with_stops([RED, BLUE]);
        for (paint_kind, draw_path) in (0..5).flat_map(|kind| {
            [false, true]
                .into_iter()
                .filter(move |&path| kind != 4 || !path)
                .map(move |path| (kind, path))
        }) {
            let draw = |scene: &mut Scene| {
                scene.set_paint_transform(Affine::new([1.0, 0.125, -0.25, 1.0, 1.25, 0.75]));
                match paint_kind {
                    0 => scene.set_paint(gradient.clone()),
                    1 => scene.set_paint(image.clone()),
                    2 => scene
                        .set_paint(Gradient::new_radial((8.0, 2.0), 8.0).with_stops([RED, BLUE])),
                    3 => scene.set_paint(
                        Gradient::new_sweep((8.0, 2.0), 0.0, core::f32::consts::TAU)
                            .with_stops([RED, BLUE]),
                    ),
                    _ => {
                        scene.set_paint(RED);
                        scene.fill_blurred_rounded_rect(
                            &Rect::new(2.0, 1.0, 8.0, 3.0),
                            1.0,
                            0.75,
                            false,
                        );
                        return;
                    }
                }
                let rect = Rect::new(0.0, 0.0, width.into(), height.into());
                if draw_path {
                    scene.fill_path(&rect.to_path(0.1));
                } else {
                    scene.fill_rect(&rect);
                }
            };
            let mut reference = Scene::new(width, height);
            draw(&mut reference);
            let expected = gpu
                .scene(&mut renderer, &mut resources, &reference)
                .unwrap();
            for magnitude in [
                65_532.0_f32,
                65_535.0,
                65_536.0,
                70_000.0,
                70_001.0,
                262_148.0,
                16_777_220.0,
            ] {
                for direction in [-1.0, 1.0] {
                    for nested in [false, true] {
                        let offset = magnitude * direction;
                        let (dx, dy) = if width > height {
                            (offset, 0.0)
                        } else {
                            (0.0, offset)
                        };
                        let mut scene = Scene::new(width, height);
                        scene.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
                            dx,
                            dy,
                        }));
                        if nested {
                            scene.push_filter_layer(Filter::from_primitive(
                                FilterPrimitive::Offset { dx: 1.0, dy: -1.0 },
                            ));
                        }
                        let extra = if nested { (1.0, -1.0) } else { (0.0, 0.0) };
                        scene.set_transform(Affine::translate((
                            -f64::from(dx) - extra.0,
                            -f64::from(dy) - extra.1,
                        )));
                        draw(&mut scene);
                        if nested {
                            scene.pop_layer();
                        }
                        scene.pop_layer();
                        let actual = gpu.scene(&mut renderer, &mut resources, &scene).unwrap();
                        if actual != expected {
                            let different =
                                actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                            mismatches.push(format!("size={width}x{height},offset=({dx},{dy}),paint={paint_kind},path={draw_path},nested={nested}: {different} differing channels"));
                        }
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn offset_layer_clip_respects_partial_rows_and_columns() {
    let gpu = Gpu::new();
    let (mut renderer, mut resources) = Renderer::new(
        &gpu.device,
        &RenderTargetConfig {
            width: 16,
            height: 8,
            format: wgpu::TextureFormat::Rgba8Unorm,
        },
    );
    let mut clip = BezPath::new();
    clip.move_to((0.5, 0.25));
    clip.line_to((9.25, 0.75));
    clip.line_to((4.5, 7.5));
    clip.close_path();
    for blend in [
        BlendMode::default(),
        BlendMode::new(Mix::Screen, Compose::SrcOver),
    ] {
        for (dx, dy) in [(1.0_f32, 1.0_f32), (3.0, 2.0), (-1.0, -1.0)] {
            let scene = |filtered: bool| {
                let mut scene = Scene::new(16, 8);
                scene.set_paint(BLUE);
                scene.fill_rect(&Rect::new(0.0, 0.0, 16.0, 8.0));
                scene.push_layer(
                    Some(&clip),
                    Some(blend),
                    None,
                    None,
                    filtered.then(|| Filter::from_primitive(FilterPrimitive::Offset { dx, dy })),
                );
                scene.set_paint(RED);
                let rect = Rect::new(0.0, 0.0, 4.0, 4.0);
                let rect = if filtered {
                    rect
                } else {
                    rect + Vec2::new(f64::from(dx), f64::from(dy))
                };
                scene.fill_path(&rect.to_path(0.1));
                scene.pop_layer();
                scene
            };
            let expected = gpu
                .scene(&mut renderer, &mut resources, &scene(false))
                .unwrap();
            assert_eq!(
                gpu.scene(&mut renderer, &mut resources, &scene(true))
                    .unwrap(),
                expected,
                "offset=({dx},{dy}),blend={blend:?}"
            );
        }
    }
}

#[test]
fn oversized_filter_textures_fail_explicitly_and_renderer_recovers() {
    let gpu = Gpu::new();
    for (width, height) in [(16, 4), (4, 16)] {
        let (mut renderer, mut resources) = Renderer::new(
            &gpu.device,
            &RenderTargetConfig {
                width,
                height,
                format: wgpu::TextureFormat::Rgba8Unorm,
            },
        );
        let mut valid = Scene::new(width, height);
        valid.set_paint(RED);
        valid.fill_rect(&Rect::new(0.0, 0.0, width.into(), height.into()));
        let expected = [255, 0, 0, 255].repeat(usize::from(width) * usize::from(height));
        for sigma in [12_000.0, 30_000.0] {
            let mut scene = Scene::new(width, height);
            scene.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: sigma,
                edge_mode: vello_common::filter_effects::EdgeMode::None,
            }));
            scene.fill_rect(&Rect::new(0.0, 0.0, width.into(), height.into()));
            scene.pop_layer();
            let error = gpu
                .scene(&mut renderer, &mut resources, &scene)
                .unwrap_err();
            assert!(
                matches!(error, RenderError::IntermediateTexture(IntermediateTextureError::TooLarge { width: w, height: h, .. }) if w.max(h) > 65535),
                "{error:?}"
            );
            assert_eq!(
                gpu.scene(&mut renderer, &mut resources, &valid).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn empty_destructive_children_use_the_filtered_sources_coordinate_space() {
    let gpu = Gpu::new();
    for (width, height) in [(16, 4), (4, 16)] {
        let (mut renderer, mut resources) = Renderer::new(
            &gpu.device,
            &RenderTargetConfig {
                width,
                height,
                format: wgpu::TextureFormat::Rgba8Unorm,
            },
        );
        for offset in [-70_000.0_f32, -70_001.0] {
            let (dx, dy) = if width > height {
                (offset, 0.0)
            } else {
                (0.0, offset)
            };
            let expected_rect = if width > height {
                Rect::new(4.0, 1.0, 8.0, 3.0)
            } else {
                Rect::new(1.0, 4.0, 3.0, 8.0)
            };
            let source = expected_rect - Vec2::new(f64::from(dx), f64::from(dy));
            for clear in [false, true] {
                let mut scene = Scene::new(width, height);
                scene.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset { dx, dy }));
                scene.set_paint(RED);
                scene.fill_rect(&source);
                if clear {
                    scene.push_layer(
                        Some(&source.to_path(0.1)),
                        Some(BlendMode::new(Mix::Normal, Compose::Clear)),
                        None,
                        None,
                        None,
                    );
                    scene.pop_layer();
                }
                scene.pop_layer();
                let pixels = gpu.scene(&mut renderer, &mut resources, &scene).unwrap();
                for (index, actual) in pixels.as_chunks::<4>().0.iter().enumerate() {
                    let x = u16::try_from(index % usize::from(width)).unwrap();
                    let y = u16::try_from(index / usize::from(width)).unwrap();
                    let inside = expected_rect.contains((f64::from(x) + 0.5, f64::from(y) + 0.5));
                    let expected = if !clear && inside {
                        [255, 0, 0, 255]
                    } else {
                        [0; 4]
                    };
                    assert_eq!(
                        *actual, expected,
                        "offset=({dx},{dy}),clear={clear},pixel=({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn disjoint_offset_children_preserve_destructive_composition() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let gpu = Gpu::new();
    let mut failures = Vec::new();
    for (nested, compose) in [false, true].into_iter().flat_map(|nested| {
        [Compose::Clear, Compose::Copy]
            .into_iter()
            .map(move |compose| (nested, compose))
    }) {
        for axis in 0..2 {
            for sign in [-1.0_f32, 1.0] {
                let case = format!("nested={nested},compose={compose:?},axis={axis},sign={sign}");
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let (mut renderer, mut resources) = Renderer::new(
                        &gpu.device,
                        &RenderTargetConfig {
                            width: 4,
                            height: 4,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                        },
                    );
                    let (base_x, base_y) = if !nested {
                        (0.0, 0.0)
                    } else if axis == 0 {
                        (70_000.0, 0.0)
                    } else {
                        (0.0, 70_000.0)
                    };
                    let mut scene = Scene::new(4, 4);
                    scene.set_paint(BLUE);
                    scene.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
                    if nested {
                        // Copy the offscreen parent back into the root after its child clears it.
                        scene.push_layer(
                            None,
                            Some(BlendMode::new(Mix::Normal, Compose::Copy)),
                            None,
                            None,
                            Some(Filter::from_primitive(FilterPrimitive::Offset {
                                dx: -base_x,
                                dy: -base_y,
                            })),
                        );
                        let parent_clip = Rect::new(
                            f64::from(base_x),
                            f64::from(base_y),
                            f64::from(base_x) + 4.0,
                            f64::from(base_y) + 4.0,
                        )
                        .to_path(0.1);
                        scene.push_layer(Some(&parent_clip), None, None, None, None);
                        scene.set_paint(RED);
                        scene.fill_rect(&Rect::new(
                            f64::from(base_x),
                            f64::from(base_y),
                            f64::from(base_x) + 4.0,
                            f64::from(base_y) + 4.0,
                        ));
                    }
                    let (dx, dy) = if axis == 0 {
                        (sign * 70_000.0, 0.0)
                    } else {
                        (0.0, sign * 70_000.0)
                    };
                    scene.push_layer(
                        None,
                        Some(BlendMode::new(Mix::Normal, compose)),
                        None,
                        None,
                        Some(Filter::from_primitive(FilterPrimitive::Offset { dx, dy })),
                    );
                    scene.set_paint(RED);
                    scene.fill_rect(&Rect::new(
                        f64::from(base_x),
                        f64::from(base_y),
                        f64::from(base_x) + 4.0,
                        f64::from(base_y) + 4.0,
                    ));
                    scene.pop_layer();
                    if nested {
                        scene.pop_layer();
                        scene.pop_layer();
                    }
                    let pixels = gpu.scene(&mut renderer, &mut resources, &scene).unwrap();
                    assert_eq!(pixels, vec![0; 4 * 4 * 4], "{case}");
                }));
                if let Err(error) = result {
                    let message = error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| {
                            error
                                .downcast_ref::<&str>()
                                .map(|message| (*message).to_owned())
                        })
                        .unwrap_or_else(|| "non-string panic".to_owned());
                    failures.push(format!("{case}: {message}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
