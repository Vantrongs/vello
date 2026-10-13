// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub(crate) mod filter_graph;
#[cfg(feature = "multithreading")]
pub(crate) mod multi_threaded;
pub(crate) mod single_threaded;

use crate::RasterizerSettings;
use crate::dispatch::filter_graph::Recording;
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use core::any::Any;
use core::fmt::Debug;
use vello_common::encode::EncodedPaint;
use vello_common::filter::FilterData;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;

pub(crate) trait Dispatcher: Any + Debug + Send {
    fn has_layers(&self) -> bool;
    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    );
    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    );
    /// Fill a pixel-aligned rectangle with the current paint.
    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    );
    /// Fill `rect` mapped by `transform`, which keeps it axis-aligned, with
    /// the current paint.
    fn fill_transformed_rect(
        &mut self,
        rect: &Rect,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        self.fill_rect_fast(
            &transform.transform_rect_bbox(*rect),
            paint,
            blend_mode,
            mask,
        );
    }
    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    );
    fn push_clip_rect(&mut self, rect: &Rect);
    fn pop_clip(&mut self);
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
    );
    fn pop_layer(&mut self);
    fn reset(&mut self, width: u16, height: u16);
    fn flush(&mut self);
    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    );
    fn is_multi_threaded(&self) -> bool;
    /// Start recording the filter layer just pushed with `filter`. Its commands
    /// go to the returned recording until it is popped, and the recording then
    /// to [`Self::draw_recording`].
    fn record_filter_layer(&mut self, filter: FilterData) -> Recording {
        let _ = filter;
        unreachable!("only the single-threaded dispatcher renders filters");
    }
    /// Composite the filter layer recorded in `recording`.
    fn draw_recording(&mut self, recording: Recording) {
        let _ = recording;
        unreachable!("only the single-threaded dispatcher renders filters");
    }
}
