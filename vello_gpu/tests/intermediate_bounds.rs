// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Device-level coverage of intermediate atlas edges across renderer reuse.

#![cfg(feature = "wgpu")]

use vello_common::color::palette::css::{BLUE, RED};
use vello_common::kurbo::{Rect, Shape};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_gpu::{
    ClearSettings, RenderSize, RenderTargetConfig, Renderer, Resources, Scene, TargetInit,
    TextureBindings,
};

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
fn odd_root_blend_does_not_leave_pixels_for_the_next_frame() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("GPU regression requires a wgpu adapter (a software Vulkan adapter is sufficient)");
    eprintln!("intermediate bounds adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("intermediate bounds regression"),
        ..Default::default()
    }))
    .expect("create intermediate bounds regression device");

    for (width, height) in [(13, 7), (7, 13)] {
        for use_depth in [false, true] {
            let config = RenderTargetConfig {
                width,
                height,
                format: wgpu::TextureFormat::Rgba8Unorm,
            };
            let (mut renderer, mut resources) = Renderer::new(&device, &config);
            let mut first = Scene::new(width, height);
            first.set_paint(RED);
            // Path strips cover complete tiles, including the partial viewport edge.
            first.fill_path(&Rect::new(-4.0, -4.0, 32.0, 32.0).to_path(0.1));
            first.set_blend_mode(BlendMode::new(Mix::Screen, Compose::SrcOver));
            first.set_paint(BLUE);
            first.fill_rect(&Rect::new(1.0, 1.0, 3.0, 3.0));
            let first_pixels = render_pixels(
                &mut renderer,
                &mut resources,
                &device,
                &queue,
                &first,
                use_depth,
            );
            for (index, pixel) in first_pixels.as_chunks::<4>().0.iter().enumerate() {
                let x = index % usize::from(width);
                let y = index / usize::from(width);
                let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                    [255, 0, 255, 255]
                } else {
                    [255, 0, 0, 255]
                };
                assert_eq!(
                    *pixel, expected,
                    "{width}x{height}, depth={use_depth}: first-frame pixel ({x}, {y})"
                );
            }

            let mut second = Scene::new(width.next_multiple_of(4), height.next_multiple_of(4));
            second.set_blend_mode(BlendMode::new(Mix::Screen, Compose::SrcOver));
            second.set_paint(BLUE);
            second.fill_rect(&Rect::new(1.0, 1.0, 3.0, 3.0));
            let actual = render_pixels(
                &mut renderer,
                &mut resources,
                &device,
                &queue,
                &second,
                use_depth,
            );
            let mismatches = actual
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
                .filter(|(index, pixel)| {
                    let x = index % usize::from(second.width());
                    let y = index / usize::from(second.width());
                    let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                        [0, 0, 255, 255]
                    } else {
                        [0, 0, 0, 0]
                    };
                    **pixel != expected
                })
                .collect::<Vec<_>>();
            assert!(
                mismatches.is_empty(),
                "{width}x{height}, depth={use_depth}: {} of {} pixels differ; first differences: {:?}",
                mismatches.len(),
                actual.len() / 4,
                &mismatches[..mismatches.len().min(8)]
            );
            eprintln!(
                "{width}x{height}, depth={use_depth}: checked {} exact RGBA pixels after renderer reuse",
                actual.len() / 4
            );
        }
    }
}
