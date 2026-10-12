// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RasterizerSettings;
use crate::coarse::CommandBucketer;
use crate::coarse::depth::DepthBuffer;
use crate::dispatch::Dispatcher;
use crate::dispatch::windowed_filter::{
    Bounds, Clip, Command, FilterOutput, Layer, Recording, WindowedFilter, identity_filter,
};
use crate::filter::context::FilterContext;
use crate::filter::pixmap::FilterPixmap;
use crate::fine::{Fine, FineKernel, FineRenderParams, FineResources, rasterize_region};
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::record::RecordedFill;
use crate::region::Regions;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::RefCell;
use vello_common::TargetInit;
use vello_common::color::AlphaColor;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::{FilterData, FilterLayerPlacement, PreparedFilter};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint, PremulColor};
use vello_common::pixmap::PixmapMut;
use vello_common::record::{
    CommandRecorder, LayerClip, LayerProps, Node, PoppedLayer, RecordedLayerKind,
};
use vello_common::strip_generator::{GenerationMode, StripStorage};
use vello_common::util::strip_bbox;
use vello_common::viewport::ViewportState;

/// Single-threaded implementation of the rendering dispatcher.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    /// Reusable coarse bucketer that converts recorded commands into per-row render commands.
    bucketer: RefCell<CommandBucketer>,
    /// Viewport state.
    viewport: ViewportState,
    /// Recorder for root and filter-layer command streams, plus layer metadata.
    recorder: CommandRecorder<RecordedFill>,
    /// Storage for generated strips and alpha coverage data.
    strip_storage: StripStorage,
    /// How each filter layer is rendered, in the order of `recorder.filter_layers`.
    filter_sources: Vec<FilterSource>,
    /// The filter layer whose source does not fit the `u16` domain and whose
    /// commands are therefore recorded instead of drawn.
    recording: Option<Box<Recording>>,
    /// Position of the root target in the space paints are encoded in.
    paint_origin: (u32, u32),
    /// Translation from root target pixels to the pixels masks are sampled in.
    mask_origin: (i64, i64),
    /// SIMD level for fearless SIMD dispatch.
    level: Level,
}

/// How a filter layer is rendered.
#[derive(Debug)]
enum FilterSource {
    /// From its own commands, drawn in a source viewport that is translated by
    /// `source_shift` from root target coordinates.
    Drawn { source_shift: (i64, i64) },
    /// In windows, see [`WindowedFilter`].
    Windowed(Box<WindowedFilter>),
    /// By the enclosing windowed filter, as its nested output `output`; the
    /// root target of this dispatcher starts at `origin` in the output's space.
    Injected { output: usize, origin: (i64, i64) },
    /// Not filtered: the layer holds the content of a window and is returned
    /// by [`SingleThreadedDispatcher::render_window`].
    WindowContent,
}

impl SingleThreadedDispatcher {
    /// Creates a new single-threaded dispatcher for the given dimensions.
    ///
    /// # Arguments
    /// * `width` - Width of the rendering surface in pixels.
    /// * `height` - Height of the rendering surface in pixels.
    /// * `level` - SIMD level to use for rasterization.
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            bucketer: RefCell::new(CommandBucketer::from_wh(width, height)),
            viewport: ViewportState::new(width, height, level),
            recorder: CommandRecorder::new(width, height),
            strip_storage: StripStorage::new(GenerationMode::Append),
            filter_sources: Vec::new(),
            recording: None,
            paint_origin: (0, 0),
            mask_origin: (0, 0),
            level,
        }
    }

    /// Create a dispatcher for a window of a windowed filter's source space
    /// and let `draw` replay the window's commands into it.
    ///
    /// Paints are encoded in that space, where the window starts at
    /// `paint_origin`; masks are sampled at window pixels moved by `mask_origin`.
    /// Everything drawn goes into one filter layer that holds the window content.
    pub(crate) fn window(
        width: u16,
        height: u16,
        level: Level,
        paint_origin: (u32, u32),
        mask_origin: (i64, i64),
        draw: impl FnOnce(&mut Self),
    ) -> Self {
        let mut window = Self {
            paint_origin,
            mask_origin,
            ..Self::new(width, height, level)
        };
        let content = identity_filter();
        window.viewport.push_root_viewport(&content);
        window.recorder.push_layer(
            LayerProps {
                blend_mode: BlendMode::default(),
                opacity: 1.0,
                mask: None,
                clip_path: None,
            },
            Some(content),
        );
        window.filter_sources.push(FilterSource::WindowContent);
        draw(&mut window);
        window.pop_layer();
        window
    }

    /// Return the bounds of the window content in the source space, where the window starts at `origin`.
    pub(crate) fn window_content(&self, origin: (i64, i64)) -> Bounds {
        Bounds::of_rect(self.window_content_placement().pixmap_bbox(), origin)
    }

    fn window_content_placement(&self) -> FilterLayerPlacement {
        let RecordedLayerKind::Filter { placement, .. } =
            &self.recorder.layers[self.recorder.filter_layers[0] as usize].kind
        else {
            unreachable!("a window starts with its content layer");
        };
        *placement
    }

    /// Render the window content, or return `None` if it is empty. The bounds
    /// of the returned pixmap are relative to the window.
    pub(crate) fn render_window<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        nested: &[FilterOutput],
    ) -> Option<(FilterPixmap, RectU16)> {
        let bbox = self.window_content_placement().pixmap_bbox();
        if bbox.is_empty() {
            return None;
        }
        let filters =
            self.rasterize_filter_layers::<S, F>(simd, encoded_paints, image_resolver, nested);
        let pixmap = self.rasterize_layer::<S, F>(
            simd,
            self.recorder.filter_layers[0],
            (0, 0),
            &filters,
            encoded_paints,
            image_resolver,
        );
        Some((pixmap, bbox))
    }

    /// Push and pop a nested filter layer whose output is rendered by the
    /// enclosing windowed filter and composited at `dest`.
    pub(crate) fn push_rendered_filter_layer(
        &mut self,
        layer: &Layer,
        to_window: Affine,
        filter: FilterData,
        dest: RectU16,
        output: usize,
        origin: (i64, i64),
    ) {
        let clip_path = layer.clip_path.as_ref().map(|path| {
            self.layer_clip(
                path,
                layer.fill_rule,
                to_window * layer.clip_transform,
                layer.aliasing_threshold,
            )
        });
        self.recorder.push_layer(
            LayerProps {
                blend_mode: layer.blend_mode,
                opacity: layer.opacity,
                mask: layer.mask.clone(),
                clip_path,
            },
            Some(filter),
        );
        self.recorder
            .pop_rendered_filter_layer(FilterLayerPlacement::rendered_at(dest));
        self.filter_sources
            .push(FilterSource::Injected { output, origin });
    }

    /// Rasterizes the scene using f32 precision (high quality).
    ///
    /// This dispatches to the appropriate SIMD implementation based on the
    /// configured level, using f32 for intermediate calculations.
    #[cfg(feature = "f32_pipeline")]
    fn rasterize_f32(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    /// Rasterizes the scene using u8 precision (fast).
    ///
    /// This dispatches to the appropriate SIMD implementation based on the
    /// configured level, using u8 for intermediate calculations to maximize speed.
    #[cfg(feature = "u8_pipeline")]
    fn rasterize_u8(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    // Note: We purposefully don't add `vectorize` to each of these helpers,
    // since vectorization is applied wherever necessary in child functions.
    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let filters =
            self.rasterize_filter_layers::<S, F>(simd, encoded_paints, image_resolver, &[]);
        let target_init = settings.target_init.map(PremulColor::from_alpha_color);
        let params = FineRenderParams {
            scene_size: (scene_width, scene_height),
            target_offset: settings.offset,
        };

        self.bucket_and_rasterize::<S, F>(
            simd,
            &self.recorder.nodes,
            RectU16::new(0, 0, scene_width, scene_height),
            &filters,
            target,
            params,
            self.mask_origin,
            target_init,
            self.recorder.root_is_blend_target,
            encoded_paints,
            image_resolver,
        );
    }

    fn bucket_and_rasterize<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        cmds: &[Node],
        viewport: RectU16,
        filter_ctx: &FilterContext,
        mut target: PixmapMut<'_>,
        params: FineRenderParams,
        mask_offset: (i64, i64),
        target_init: TargetInit<PremulColor>,
        root_is_blend_target: bool,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let mut bucketer = self.bucketer.borrow_mut();
        bucketer.reset(viewport);
        bucketer.bucket_commands(
            cmds,
            &self.recorder.draws,
            &self.recorder.layers,
            &self.strip_storage.strips,
            encoded_paints,
            filter_ctx,
        );

        let alpha_buffers = &[self.strip_storage.alphas.as_slice()];
        let resources = FineResources {
            alpha_buffers,
            encoded_paints,
            filter_paints: &bucketer.filter_paints,
            image_resolver,
        };
        let mut regions = Regions::new(
            &mut target,
            params.scene_size,
            params.target_offset,
            bucketer.rows().len(),
        );
        Self::rasterize_target::<S, F>(
            simd,
            &bucketer,
            resources,
            &mut regions,
            mask_offset,
            self.paint_origin,
            target_init,
            root_is_blend_target,
        );
    }

    fn rasterize_target<S: Simd, F: FineKernel<S>>(
        simd: S,
        bucketer: &CommandBucketer,
        resources: FineResources<'_>,
        regions: &mut Regions<'_>,
        mask_offset: (i64, i64),
        paint_origin: (u32, u32),
        target_init: TargetInit<PremulColor>,
        root_is_blend_target: bool,
    ) {
        // TODO: Reuse fine and depth buffer across targets?
        let mut fine = Fine::<S, F>::new(simd, bucketer.width());
        fine.set_mask_offset(mask_offset);
        fine.set_paint_origin(paint_origin);
        let mut depth = DepthBuffer::new(bucketer.width());
        regions.update(|region| {
            rasterize_region::<S, F>(
                &mut fine,
                &mut depth,
                region,
                bucketer,
                resources,
                target_init,
                root_is_blend_target,
            );
        });
    }

    fn record_fill(
        &mut self,
        strip_start: usize,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        let strip_end = self.strip_storage.strips.len();
        let strip_range = strip_start..strip_end;
        let strips = &self.strip_storage.strips[strip_range.start..strip_range.end];
        let draw = RecordedFill::new(0, strip_range, paint, blend_mode, mask);

        self.recorder.push_draw(draw, strips);
    }

    /// Generate the strips of a layer clip in the active viewport.
    fn layer_clip(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) -> LayerClip {
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, existing_clip| {
                strip_generator.generate_filled_path(
                    path,
                    fill_rule,
                    transform,
                    aliasing_threshold,
                    strip_storage,
                    existing_clip.map(|clip| clip.path),
                );

                let strip_range = strip_start..strip_storage.strips.len();
                LayerClip {
                    bbox: strip_bbox(&strip_storage.strips[strip_range.clone()])
                        .unwrap_or(RectU16::ZERO),
                    strip_range,
                    thread_idx: 0,
                }
            })
    }

    /// Render and filter every filter layer. `nested` holds the outputs that an
    /// enclosing windowed filter rendered for the layers injected into a window.
    fn rasterize_filter_layers<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        nested: &[FilterOutput],
    ) -> FilterContext {
        // TODO: Reuse across frames so that pixmaps can be reused.
        let mut filter_ctx = FilterContext::new(self.recorder.layers.len());
        // We record filter layers upon "push", so nested filter layers get added after their
        // parents. Subsequent sibling filter layers also get added after the previous layer they
        // are composited into. Therefore, iterating in reverse order is enough to ensure that all
        // dependencies have been rendered before they are invoked.
        for (&id, source) in self
            .recorder
            .filter_layers
            .iter()
            .zip(&self.filter_sources)
            .rev()
        {
            let RecordedLayerKind::Filter {
                filter_data,
                placement,
            } = &self.recorder.layers[id as usize].kind
            else {
                unreachable!("filter_layers only contains filter layers");
            };
            if placement.pixmap_bbox().is_empty() {
                continue;
            }

            let pixmap = match source {
                FilterSource::Drawn { source_shift } => {
                    let mut pixmap = self.rasterize_layer::<S, F>(
                        simd,
                        id,
                        *source_shift,
                        &filter_ctx,
                        encoded_paints,
                        image_resolver,
                    );
                    F::filter_layer(
                        &mut pixmap,
                        PreparedFilter::new(&filter_data.filter, &filter_data.transform),
                        filter_ctx.scratch(),
                    );
                    pixmap.into_pixmap()
                }
                FilterSource::Windowed(filter) => filter
                    .render::<S, F>(simd, encoded_paints, image_resolver)
                    .crop(placement.dest_bbox(), (0, 0)),
                FilterSource::Injected { output, origin } => {
                    nested[*output].crop(placement.dest_bbox(), *origin)
                }
                FilterSource::WindowContent => continue,
            };

            // Save the filtered pixmap to disk for debugging.
            // #[cfg(all(debug_assertions, feature = "std", feature = "png"))]
            // save_filtered_layer_debug(&pixmap, id);

            filter_ctx.set_layer(id as usize, pixmap);
        }

        filter_ctx
    }

    /// Rasterize the commands of filter layer `id` into its pixmap, unfiltered.
    fn rasterize_layer<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        id: u32,
        source_shift: (i64, i64),
        filter_ctx: &FilterContext,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> FilterPixmap {
        let layer = &self.recorder.layers[id as usize];
        let pixmap_bbox = layer.bbox;
        let (width, height) = (pixmap_bbox.width(), pixmap_bbox.height());
        // TODO: See https://github.com/linebender/vello/pull/1701#discussion_r3400709986, explore
        // using pools for more resources.
        let mut pixmap = FilterPixmap::new(u32::from(width), u32::from(height));
        let params = FineRenderParams {
            scene_size: (width, height),
            target_offset: (0, 0),
        };
        let mask_offset = (
            i64::from(pixmap_bbox.x0) - source_shift.0 + self.mask_origin.0,
            i64::from(pixmap_bbox.y0) - source_shift.1 + self.mask_origin.1,
        );

        self.bucket_and_rasterize::<S, F>(
            simd,
            &layer.nodes,
            pixmap_bbox,
            filter_ctx,
            pixmap.as_target(),
            params,
            mask_offset,
            TargetInit::Clear(PremulColor::from_alpha_color(AlphaColor::TRANSPARENT)),
            false,
            encoded_paints,
            image_resolver,
        );
        pixmap
    }

    /// Start recording a filter layer whose source does not fit the `u16` domain.
    fn start_recording(&self, filter: &FilterData) -> Recording {
        Recording::new(
            filter.clone(),
            Bounds::of_viewport(self.viewport.width(), self.viewport.height()),
            self.viewport.source_shift(),
            self.viewport.clips().map(Clip::from_pushed).collect(),
        )
    }

    /// Replay a finished recording into windows and composite the visible part of its output.
    fn finish_recording(&mut self, recording: Recording) {
        let filter = recording.finish(self.level);
        let view = Bounds::of_viewport(self.viewport.width(), self.viewport.height());
        self.recorder
            .pop_rendered_filter_layer(FilterLayerPlacement::rendered_at(
                filter.placement_in(view),
            ));
        self.filter_sources
            .push(FilterSource::Windowed(Box::new(filter)));
    }
}

impl Dispatcher for SingleThreadedDispatcher {
    fn has_layers(&self) -> bool {
        self.recorder.has_layers()
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        if let Some(recording) = &mut self.recording {
            recording.record(Command::FillPath {
                path: path.clone(),
                fill_rule,
                transform,
                paint,
                blend_mode,
                aliasing_threshold,
                mask,
            });
            return;
        }
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, clip_path| {
                strip_generator.generate_filled_path(
                    path,
                    fill_rule,
                    transform,
                    aliasing_threshold,
                    strip_storage,
                    clip_path.map(|clip| clip.path),
                );
            });
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        if let Some(recording) = &mut self.recording {
            recording.record(Command::StrokePath {
                path: path.clone(),
                stroke: stroke.clone(),
                transform,
                paint,
                blend_mode,
                aliasing_threshold,
                mask,
            });
            return;
        }
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, clip_path| {
                strip_generator.generate_stroked_path(
                    path,
                    stroke,
                    transform,
                    aliasing_threshold,
                    strip_storage,
                    clip_path.map(|clip| clip.path),
                );
            });
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        if let Some(recording) = &mut self.recording {
            recording.record(Command::FillRect {
                rect: *rect,
                paint,
                blend_mode,
                mask,
            });
            return;
        }
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, clip_path| {
                strip_generator.generate_filled_rect_fast(rect, strip_storage, clip_path);
            });
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter_data: Option<FilterData>,
    ) {
        if let Some(recording) = &mut self.recording {
            let layer = Layer {
                clip_path: clip_path.cloned(),
                fill_rule,
                clip_transform,
                blend_mode,
                opacity,
                aliasing_threshold,
                mask,
                filter: None,
            };
            recording.push_layer(layer, filter_data);
            return;
        }

        let fits = |filter: &FilterData| {
            filter
                .source_viewport_size(self.viewport.width(), self.viewport.height())
                .is_some()
        };
        let recording = filter_data
            .as_ref()
            .filter(|filter| !fits(filter))
            .map(|filter| self.start_recording(filter));
        // The clip applies where the filtered layer is composited, in the parent viewport.
        let clip_path = clip_path.map(|clip_path| {
            self.layer_clip(clip_path, fill_rule, clip_transform, aliasing_threshold)
        });
        if let Some(filter_data) = &filter_data
            && recording.is_none()
        {
            self.viewport.push_root_viewport(filter_data);
            self.filter_sources.push(FilterSource::Drawn {
                source_shift: self.viewport.source_shift(),
            });
        }
        self.recording = recording.map(Box::new);

        self.recorder.push_layer(
            LayerProps {
                blend_mode,
                opacity,
                mask,
                clip_path,
            },
            filter_data,
        );
    }

    fn pop_layer(&mut self) {
        if let Some(recording) = &mut self.recording {
            if recording.pop_layer() {
                return;
            }
            let recording = self.recording.take().unwrap();
            self.finish_recording(*recording);
            return;
        }
        match self.recorder.pop_layer() {
            PoppedLayer::Regular => {}
            PoppedLayer::Filter => {
                self.viewport.pop_root_viewport();
            }
        }
    }

    fn reset(&mut self, width: u16, height: u16) {
        // Bucketer will be reset on demand, so no need to reset it here.
        self.recorder.reset(width, height);
        self.strip_storage.clear();
        self.filter_sources.clear();
        self.recording = None;
        self.viewport.reset(width, height);
    }

    fn flush(&mut self) {
        // No-op for single-threaded dispatcher (no work queue to flush).
    }

    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        // If only the u8 pipeline is enabled, then use it.
        #[cfg(all(feature = "u8_pipeline", not(feature = "f32_pipeline")))]
        {
            self.rasterize_u8(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        // If only the f32 pipeline is enabled, then use it.
        #[cfg(all(feature = "f32_pipeline", not(feature = "u8_pipeline")))]
        {
            self.rasterize_f32(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        // If both pipelines are enabled, select precision based on render mode parameter.
        #[cfg(all(feature = "u8_pipeline", feature = "f32_pipeline"))]
        match settings.render_mode {
            crate::RenderMode::OptimizeSpeed => {
                // Use u8 precision for faster rendering.
                self.rasterize_u8(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
            crate::RenderMode::OptimizeQuality => {
                // Use f32 precision for higher quality.
                self.rasterize_f32(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
        }

        #[cfg(all(not(feature = "u8_pipeline"), not(feature = "f32_pipeline")))]
        {
            // This case never gets hit because there is a compile_error in the root.
            // But have this code disables some warnings and makes the compile error easier to read
            let _ = (
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        // Clips outlive the layers they are pushed in, so a recorded clip also
        // applies in the parent.
        if let Some(recording) = &mut self.recording {
            recording.record(Command::PushClip(Clip::Path {
                path: path.clone(),
                fill_rule,
                transform,
                aliasing_threshold,
            }));
        }
        self.viewport
            .push_clip_path(path, fill_rule, transform, aliasing_threshold);
    }

    fn push_clip_rect(&mut self, rect: &Rect) {
        if let Some(recording) = &mut self.recording {
            recording.record(Command::PushClip(Clip::Rect(*rect)));
        }
        self.viewport.push_clip_rect(rect);
    }

    fn pop_clip(&mut self) {
        if let Some(recording) = &mut self.recording {
            recording.record(Command::PopClip);
        }
        self.viewport.pop_clip();
    }

    fn is_multi_threaded(&self) -> bool {
        false
    }
}

/// Saves a filtered pixmap to disk for debugging purposes.
/// Only available in debug builds with `std` and `png` features enabled.
#[allow(
    dead_code,
    reason = "useful debug utility, can be enabled by uncommenting the call site"
)]
#[cfg(all(debug_assertions, feature = "std", feature = "png"))]
fn save_filtered_layer_debug(pixmap: &vello_common::pixmap::Pixmap, layer_id: usize) {
    use std::path::PathBuf;

    let diffs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_tests/diffs");
    let _ = std::fs::create_dir_all(&diffs_path);
    let filename = diffs_path.join(alloc::format!("filtered_layer_{layer_id}.png"));

    if let Ok(png_data) = pixmap.clone().into_png() {
        let _ = std::fs::write(&filename, &png_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::Shape;
    use vello_common::color::palette::css::BLUE;
    use vello_common::paint::PremulColor;

    /// Verifies that `reset()` properly clears all internal buffers and state.
    ///
    /// This is important to ensure that a dispatcher can be reused for multiple
    /// rendering passes without accumulating stale data from previous frames.
    #[test]
    fn buffers_cleared_on_reset() {
        let mut dispatcher = SingleThreadedDispatcher::new(100, 100, Level::new());

        // Render a simple shape to populate internal buffers.
        dispatcher.fill_path(
            &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
            BlendMode::default(),
            None,
            None,
        );

        // Ensure there is data to clear.
        assert!(!dispatcher.strip_storage.strips.is_empty());
        assert!(!dispatcher.recorder.nodes.is_empty());

        dispatcher.reset(100, 100);

        // Verify all buffers are cleared.
        assert!(dispatcher.strip_storage.strips.is_empty());
        assert!(dispatcher.strip_storage.alphas.is_empty());
        assert!(dispatcher.recorder.nodes.is_empty());
        assert!(dispatcher.recorder.layers.is_empty());
        assert!(!dispatcher.viewport.has_root_viewports());
    }
}
