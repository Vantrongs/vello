// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// Error-free addition, apart from underflow of the roundoff term.
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    (sum, (a - (sum - b_virtual)) + (b - b_virtual))
}

/// Grow a nonoverlapping expansion with error-free `TwoSum`. The highest component
/// gives the sign of the exact sum, even after cancellation. Each input adds at
/// most one component.
pub(crate) struct Expansion<const N: usize> {
    parts: [f64; N],
    len: usize,
}

impl<const N: usize> Expansion<N> {
    pub(crate) fn new() -> Self {
        Self {
            parts: [0.; N],
            len: 0,
        }
    }

    pub(crate) fn add(&mut self, mut value: f64) {
        let mut used = 0;
        for i in 0..self.len {
            let part = self.parts[i];
            let (sum, error) = two_sum(value, part);
            if error != 0.0 {
                self.parts[used] = error;
                used += 1;
            }
            value = sum;
        }
        if value != 0.0 || used == 0 {
            self.parts[used] = value;
            used += 1;
        }
        self.len = used;
    }

    pub(crate) fn leading(&self) -> f64 {
        self.parts[self.len - 1]
    }

    pub(crate) fn estimate(&self) -> f64 {
        self.parts[..self.len].iter().sum()
    }
}
