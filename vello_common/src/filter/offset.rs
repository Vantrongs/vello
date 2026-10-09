// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The offset filter.

/// Translation/shift filter.
///
/// This shifts the input image by `(dx, dy)` in device pixel space.
#[derive(Clone, Copy, Debug)]
pub struct Offset {
    /// The x-offset that should be applied.
    pub dx: f32,
    /// The y-offset that should be applied.
    pub dy: f32,
}

impl Offset {
    /// Create a new offset filter.
    pub fn new(dx: f32, dy: f32) -> Self {
        super::validate_offset(dx, dy);
        Self { dx, dy }
    }
}

#[cfg(test)]
mod tests {
    use super::Offset;

    #[test]
    fn nonfinite_offsets_are_rejected() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| Offset::new(value, 0.0)).is_err());
            assert!(std::panic::catch_unwind(|| Offset::new(0.0, value)).is_err());
        }
        let offset = Offset::new(-70000.5, 70000.5);
        assert_eq!((offset.dx, offset.dy), (-70000.5, 70000.5));
    }
}
