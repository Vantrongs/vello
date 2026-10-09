// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::filter::pixmap::FilterPixmap;
use alloc::sync::Arc;
use alloc::vec::Vec;

#[derive(Debug, Default)]
pub(crate) struct FilterContext {
    /// The rendered pixmaps for each filter layer.
    layers: Vec<Option<Arc<FilterPixmap>>>,
    scratch: ScratchBuffer,
}

impl FilterContext {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            scratch: ScratchBuffer::new(),
        }
    }

    pub(crate) fn scratch(&mut self) -> &mut ScratchBuffer {
        &mut self.scratch
    }

    pub(crate) fn set_layer(&mut self, id: usize, pixmap: FilterPixmap) {
        if id >= self.layers.len() {
            self.layers.resize_with(id + 1, || None);
        }
        self.layers[id] = Some(Arc::new(pixmap));
    }

    pub(crate) fn filter_layer(&self, id: usize) -> Option<Arc<FilterPixmap>> {
        self.layers.get(id).and_then(Option::as_ref).cloned()
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScratchBuffer {
    scratch_buffer: Option<FilterPixmap>,
}

impl ScratchBuffer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_scratch_buffer(&mut self, width: u32, height: u32) -> &mut FilterPixmap {
        match &mut self.scratch_buffer {
            None => {
                self.scratch_buffer = Some(FilterPixmap::new(width, height));
            }
            Some(buf) if buf.width() < width || buf.height() < height => {
                buf.resize(width, height);
            }
            Some(_) => {}
        }

        self.scratch_buffer.as_mut().unwrap()
    }
}
