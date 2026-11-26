// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocation metering during directed-edge preparation.

use alloc::vec::Vec;
use core::cell::Cell;

/// Cumulative allowance for additional Vello-owned vector capacity during path preparation.
/// Reused capacity is free. Kurbo's private stroke outline buffers and rasterization after
/// submission are outside this allowance.
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
pub(crate) struct Meter<'a>(pub(crate) Option<&'a PathBudget>);

impl Meter<'_> {
    pub(crate) fn failed(self) -> bool {
        self.0.is_some_and(|b| b.exhausted.get())
    }

    #[inline]
    pub(crate) fn room<T>(self, buf: &mut Vec<T>, extra: usize) -> bool {
        let Some(b) = self.0 else { return true };
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
            .min(buf.capacity().saturating_add(available));
        if cap < required {
            b.exhausted.set(true);
            return false;
        }
        let bytes = (cap - buf.capacity()) * size_of::<T>();
        b.remaining.set(b.remaining.get() - bytes);
        if buf.try_reserve_exact(cap - buf.len()).is_err() {
            b.exhausted.set(true);
            return false;
        }
        true
    }
}
