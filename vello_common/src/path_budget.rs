// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocation metering during directed-edge preparation.

pub use crate::cull::dash_work;
use alloc::vec::Vec;
use core::cell::Cell;

/// Allowance for Vello vector growth while preparing edges and submitting strips.
/// Reused storage is free. A growth must also fit while its old allocation is live.
#[derive(Debug)]
pub struct PathBudget {
    remaining: Cell<usize>,
    exhausted: Cell<bool>,
}

impl PathBudget {
    /// Allow up to this many bytes of additional capacity.
    pub fn new(bytes: usize) -> Self {
        Self {
            remaining: Cell::new(bytes),
            exhausted: Cell::new(false),
        }
    }

    /// Capacity allowance still available.
    pub fn remaining(&self) -> usize {
        self.remaining.get()
    }
}

/// Preparation could not reserve its next batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityExceeded;

impl core::fmt::Display for CapacityExceeded {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("could not reserve path preparation capacity")
    }
}

impl core::error::Error for CapacityExceeded {}

#[derive(Clone, Copy, Default, Debug)]
pub(crate) enum Meter<'a> {
    #[default]
    Unlimited,
    Raster(&'a PathBudget),
    Indexed(&'a PathBudget),
}

impl Meter<'_> {
    pub(crate) fn failed(self) -> bool {
        match self {
            Self::Unlimited => false,
            Self::Raster(b) | Self::Indexed(b) => b.exhausted.get(),
        }
    }

    #[inline]
    pub(crate) fn room<T>(self, buf: &mut Vec<T>, extra: usize) -> bool {
        let (Self::Raster(b) | Self::Indexed(b)) = self else {
            return true;
        };
        if b.exhausted.get() {
            return false;
        }
        if extra <= buf.capacity() - buf.len() {
            return true;
        }
        let required = buf.len().saturating_add(extra);
        let available = b.remaining.get() / size_of::<T>();
        let cap = required
            .max(buf.capacity().saturating_mul(2).max(4))
            .min(available);
        if cap < required {
            b.exhausted.set(true);
            return false;
        }
        let bytes = (cap - buf.capacity()) * size_of::<T>();
        if buf.try_reserve_exact(cap - buf.len()).is_err() {
            b.exhausted.set(true);
            return false;
        }
        b.remaining.set(b.remaining.get() - bytes);
        true
    }
}
