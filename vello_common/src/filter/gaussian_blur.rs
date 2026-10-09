// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The gaussian blur filter.

use alloc::vec::Vec;

use crate::filter_effects::EdgeMode;
use crate::kurbo::Affine;
use crate::util::extract_scales;
use core::f32::consts::E;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// Scale a blur's standard deviation uniformly based on the transformation.
///
/// Extracts the scale factors from the transformation matrix using SVD and
/// averages them to get a uniform scale factor for the blur radius.
///
/// # Arguments
/// * `std_deviation` - The blur standard deviation in user space
/// * `transform` - The transformation matrix to extract scale from
///
/// # Returns
/// The scaled standard deviation in device space
pub(crate) fn transform_blur_params(std_deviation: f32, transform: &Affine) -> f32 {
    let (scale_x, scale_y) = extract_scales(transform);
    let uniform_scale = (scale_x + scale_y) / 2.0;
    // Retain existing f32 rounding where the generic SVD helper did not clamp a
    // singular value or overflow. Blur does not divide by these scales, so its
    // degenerate transforms must not inherit the helper's nonzero floor.
    let scaled = if scale_x > 1e-6 && scale_y > 1e-6 && uniform_scale.is_finite() {
        std_deviation * uniform_scale
    } else {
        let [a, b, c, d, _, _] = transform.as_coeffs();
        let magnitude = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
        if magnitude == 0.0 {
            0.0
        } else {
            let [a, b, c, d] = [a, b, c, d].map(|value| value / magnitude);
            // For a 2x2 matrix the sum of singular values equals
            // max(hypot(a+d, b-c), hypot(a-d, b+c)). Normalize first so finite
            // coefficients cannot overflow or underflow while being squared.
            let average = (a + d).hypot(b - c).max((a - d).hypot(b + c)) / 2.0;
            (f64::from(std_deviation) * magnitude * average) as f32
        }
    };
    assert!(
        scaled.is_finite(),
        "transformed blur standard deviation must be finite"
    );
    // TODO: Support separate standard deviations along the transformed axes.
    scaled
}

/// Maximum size of the Gaussian kernel (must be odd and equal to or smaller than [`u8::MAX`]).
///
/// The multi-scale decimation algorithm guarantees that kernel size never exceeds this value.
/// Decimation stops when remaining variance ≤ 4.0 (σ ≤ 2.0), which produces kernels of size
/// at most 13 (radius = ceil(3σ) = 6, size = 1 + 2×6 = 13).
// Keep in sync with MAX_KERNEL_SIZE in vello_gpu_shaders/shaders/filter.wesl
pub const MAX_KERNEL_SIZE: usize = 13;

#[cfg(test)]
const _: () = const {
    if MAX_KERNEL_SIZE.is_multiple_of(2) {
        panic!("`MAX_KERNEL_SIZE` must be odd");
    }
    if MAX_KERNEL_SIZE > u8::MAX as usize {
        panic!("`MAX_KERNEL_SIZE` must be less than or equal to `u8::MAX`");
    }
};

/// A gaussian blur.
#[derive(Debug)]
pub struct GaussianBlur {
    /// The standard deviation.
    pub std_deviation: f32,
    /// Number of 2× decimation levels to use (0 means no decimation, direct convolution).
    pub n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    pub kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (rest is padding up to `MAX_KERNEL_SIZE`).
    pub kernel_size: u8,
    /// Edge mode for handling out-of-bounds sampling.
    pub edge_mode: EdgeMode,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter with the specified standard deviation.
    ///
    /// This precomputes the decimation plan, kernel, and radius for optimal performance.
    ///
    /// # Panics
    ///
    /// Panics if `std_deviation` is not finite.
    pub fn new(std_deviation: f32, edge_mode: EdgeMode) -> Self {
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(std_deviation);

        Self {
            std_deviation,
            edge_mode,
            n_decimations,
            kernel,
            kernel_size,
        }
    }
}

/// Compute the blur execution plan based on standard deviation.
///
/// Returns (`n_decimations`, `kernel`, `kernel_size`):
/// - `n_decimations`: Number of 2× downsampling steps to perform (per axis)
/// - `kernel`: Pre-computed Gaussian kernel weights (fixed-size array)
/// - `kernel_size`: Actual length of the kernel (rest is zero-padded)
///
/// # Panics
///
/// Panics if `std_deviation` is not finite.
pub fn plan_decimated_blur(std_deviation: f32) -> (usize, [f32; MAX_KERNEL_SIZE], u8) {
    assert!(
        std_deviation.is_finite(),
        "blur standard deviation must be finite"
    );
    if std_deviation <= 0.0 {
        // Invalid standard deviation, return identity kernel (no blur)
        let mut kernel = [0.0; MAX_KERNEL_SIZE];
        kernel[0] = 1.0;
        return (0, kernel, 1);
    }

    // Compute decimation plan using variance analysis.
    // Variance (σ²) has the additive property: applying two blurs sequentially
    // adds their variances together. We use this to decompose the blur.
    //
    // Mathematical Foundation: From probability theory, convolving two Gaussians
    // G(σ₁) ⊗ G(σ₂) = G(√(σ₁² + σ₂²)). This means variance is additive: σ²_total = σ²_1 + σ²_2.
    // Rearranging: σ²_2 = σ²_total - σ²_1, allowing us to decompose the target blur.
    let variance = std_deviation * std_deviation;
    let mut n_decimations = 0;
    // Preserve f32 rounding for existing blur plans; only widen when squaring a
    // finite standard deviation overflows f32.
    let mut remaining_variance = if variance.is_finite() {
        f64::from(variance)
    } else {
        f64::from(std_deviation).powi(2)
    };

    // Each decimation level blurs the image *twice* over the full round trip, and both passes
    // must be subtracted from the budget so the final result matches the target σ:
    // 1. The downscale applies a [1,3,3,1]/8 binomial filter (variance 0.75 in the current grid).
    // 2. The matching upscale reconstruction adds 0.75 variance too: each output samples
    //    neighbouring decimated pixels 0.5 and 1.5 original-grid pixels away from its centre,
    //    so 0.75*(0.5²) + 0.25*(1.5²) = 0.75.
    // So a level removes 0.75 + 0.75 = 1.5 of variance (in current-grid units) before the 2×
    // downsampling rescales the remaining variance by 0.25 (= 1/2²) into the next grid.
    while remaining_variance > 4.0 {
        remaining_variance = if variance.is_finite() {
            f64::from((remaining_variance as f32 - 1.5) * 0.25)
        } else {
            (remaining_variance - 1.5) * 0.25
        };
        n_decimations += 1;
    }
    // Compute the reduced standard deviation to apply at the decimated resolution
    let remaining_sigma = (remaining_variance as f32).sqrt();
    // Compute Gaussian kernel for the reduced blur
    let (kernel, kernel_size) = compute_gaussian_kernel(remaining_sigma);

    (n_decimations, kernel, kernel_size)
}

/// Compute 1D Gaussian kernel weights for separable convolution.
///
/// Returns (`kernel_weights`, `kernel_size`) where `kernel_size = 2×radius + 1`.
/// The kernel is stored in a fixed-size array to avoid heap allocation.
/// Uses the standard Gaussian formula: G(x) = exp(-x² / (2σ²)), normalized to sum to 1.
///
/// Nonpositive standard deviations produce the identity kernel.
///
/// # Panics
///
/// Panics if `std_deviation` is not finite.
pub fn compute_gaussian_kernel(std_deviation: f32) -> ([f32; MAX_KERNEL_SIZE], u8) {
    assert!(
        std_deviation.is_finite(),
        "blur standard deviation must be finite"
    );
    if std_deviation <= 0.0 || std_deviation * std_deviation == 0.0 {
        let mut kernel = [0.0; MAX_KERNEL_SIZE];
        kernel[0] = 1.0;
        return (kernel, 1);
    }
    // Use radius = 3σ to capture 99.7% of the Gaussian distribution.
    // Beyond ±3σ, the Gaussian values are negligible (<0.3%).
    let radius = (3.0 * std_deviation)
        .ceil()
        .min((MAX_KERNEL_SIZE / 2) as f32) as usize;
    let kernel_size = (1 + radius * 2) as u8;

    let mut kernel = [0.0; MAX_KERNEL_SIZE];
    // Compute Gaussian weights using the formula: G(x) = exp(-x² / (2σ²))
    // This creates a symmetric bell curve centered at the middle of the kernel.
    let gaussian_denominator = 2.0 * std_deviation * std_deviation;
    let mut sum = 0.0;
    let kernel_center = (kernel_size / 2) as f32;
    for (i, weight) in kernel.iter_mut().enumerate().take(usize::from(kernel_size)) {
        // Compute distance from center (0 at center, increases outward)
        let x = (i as f32) - kernel_center;
        // Apply Gaussian formula: weight decreases exponentially with squared distance
        *weight = E.powf(-x * x / gaussian_denominator);
        sum += *weight;
    }

    // Normalize weights to sum to 1.0, ensuring the blur doesn't change overall brightness.
    // Without normalization, blurring a uniform gray area could make it brighter/darker.
    let scale = 1.0 / sum;
    for weight in kernel.iter_mut().take(usize::from(kernel_size)) {
        *weight *= scale;
    }

    (kernel, kernel_size)
}

/// Tracks dimensions through a chain of downscale/upscale operations.
#[derive(Debug, Default)]
pub struct DecimationSizer {
    width: u32,
    height: u32,
    dim_stack: Vec<(u32, u32)>,
}

impl DecimationSizer {
    /// Create a new sizer with the given initial dimensions.
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            dim_stack: Vec::new(),
        }
    }

    /// Reset the sizer so it can be reused.
    #[inline]
    pub fn reset(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.dim_stack.clear();
    }

    /// Returns the current logical dimensions.
    #[inline]
    pub fn current(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Apply a new downscale operation.
    #[inline]
    pub fn downscale(&mut self) -> (u32, u32) {
        self.dim_stack.push((self.width, self.height));
        self.width = self.width.div_ceil(2);
        self.height = self.height.div_ceil(2);
        (self.width, self.height)
    }

    /// Apply a new upscale operation.
    #[inline]
    pub fn upscale(&mut self) -> (u32, u32) {
        let (target_w, target_h) = self.dim_stack.pop().unwrap();
        self.width = target_w;
        self.height = target_h;
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use crate::filter::gaussian_blur::{
        DecimationSizer, compute_gaussian_kernel, plan_decimated_blur,
    };

    /// Test Gaussian kernel computation for small σ.
    #[test]
    fn test_gaussian_kernel_small_sigma() {
        let (kernel, size) = compute_gaussian_kernel(1.0);
        // For σ=1.0, radius = ceil(3.0) = 3, size = 2*3+1 = 7
        assert_eq!(size, 7);

        // Kernel should be symmetric
        for i in 0..size / 2 {
            assert!((kernel[usize::from(i)] - kernel[usize::from(size - 1 - i)]).abs() < 1e-6);
        }

        // Kernel should sum to 1.0 (normalized)
        let sum: f32 = kernel.iter().take(usize::from(size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Center should be the largest weight
        let center_idx = size / 2;
        for i in 0..size {
            if i != center_idx {
                assert!(kernel[usize::from(center_idx)] >= kernel[usize::from(i)]);
            }
        }
    }

    /// Test Gaussian kernel computation for very small σ (near-zero).
    #[test]
    fn test_gaussian_kernel_very_small_sigma() {
        let (kernel, size) = compute_gaussian_kernel(0.1);
        // For σ=0.1, radius = ceil(0.3) = 1, size = 3
        assert_eq!(size, 3);
        // Should sum to 1.0
        let sum: f32 = kernel.iter().take(usize::from(size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Center weight should be dominant for very small σ
        assert!(kernel[1] > 0.9); // Center is highly weighted
    }

    /// Test Gaussian kernel for fractional σ.
    #[test]
    fn test_gaussian_kernel_fractional_sigma() {
        let (kernel, size) = compute_gaussian_kernel(0.5);
        // For σ=0.5, radius = ceil(1.5) = 2, size = 5
        assert_eq!(size, 5);

        // Should still sum to 1.0
        let sum: f32 = kernel.iter().take(usize::from(size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    /// Test decimation plan for small blur (no decimation).
    #[test]
    fn test_plan_no_decimation() {
        let (n_decimations, _kernel, _size) = plan_decimated_blur(1.0);
        // σ=1.0 → variance=1.0, should not decimate
        assert_eq!(n_decimations, 0);
    }

    /// Test decimation plan for medium blur (some decimation).
    #[test]
    fn test_plan_with_decimation() {
        let (n_decimations, _kernel, _size) = plan_decimated_blur(5.0);
        // σ=5.0 → variance=25.0, should decimate
        assert_eq!(n_decimations, 2);
    }

    /// Test decimation plan at boundary (σ=2.0).
    #[test]
    fn test_plan_decimation_boundary() {
        let (n_decimations, _kernel, _size) = plan_decimated_blur(2.0);
        // σ=2.0 → variance=4.0, right at the boundary
        assert_eq!(n_decimations, 0);
    }

    /// Test decimation plan for negative σ (invalid, should return identity).
    #[test]
    fn test_plan_negative_sigma() {
        let (n_decimations, kernel, size) = plan_decimated_blur(-1.0);
        assert_eq!(n_decimations, 0);
        assert_eq!(size, 1);
        assert!((kernel[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_decimation_sizer_even() {
        let mut sizer = DecimationSizer::new(8, 8);
        assert_eq!(sizer.current(), (8, 8));

        assert_eq!(sizer.downscale(), (4, 4));
        assert_eq!(sizer.downscale(), (2, 2));

        assert_eq!(sizer.upscale(), (4, 4));
        assert_eq!(sizer.upscale(), (8, 8));
    }

    #[test]
    fn test_decimation_sizer_odd() {
        let mut sizer = DecimationSizer::new(5, 7);
        assert_eq!(sizer.downscale(), (3, 4));
        assert_eq!(sizer.downscale(), (2, 2));

        // Upscale clamps to the pre-downscale target
        assert_eq!(sizer.upscale(), (3, 4));
        assert_eq!(sizer.upscale(), (5, 7));
    }

    #[test]
    fn test_decimation_sizer_single_level() {
        let mut sizer = DecimationSizer::new(100, 50);
        assert_eq!(sizer.downscale(), (50, 25));
        assert_eq!(sizer.upscale(), (100, 50));
    }

    #[test]
    fn decimation_restores_exact_odd_and_maximum_dimensions() {
        for (width, height) in [(5, 3), (65_535, 1), (1, 65_537), (u32::MAX, u32::MAX)] {
            let mut sizer = DecimationSizer::new(width, height);
            for _ in 0..4 {
                sizer.downscale();
            }
            for _ in 0..4 {
                sizer.upscale();
            }
            assert_eq!(sizer.current(), (width, height));
        }
    }

    #[test]
    fn finite_extreme_sigmas_produce_finite_bounded_plans() {
        for sigma in [f32::from_bits(1), f32::MIN_POSITIVE, 1.0e20, f32::MAX] {
            let (levels, kernel, count) = plan_decimated_blur(sigma);
            assert!(levels <= 128);
            assert!(count > 0 && usize::from(count) <= super::MAX_KERNEL_SIZE);
            assert!(kernel.iter().all(|value| value.is_finite()));
            assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
            let (kernel, _) = compute_gaussian_kernel(sigma);
            assert!(kernel.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn nonfinite_sigmas_fail_before_planning() {
        for sigma in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| plan_decimated_blur(sigma)).is_err());
            assert!(std::panic::catch_unwind(|| compute_gaussian_kernel(sigma)).is_err());
        }
    }

    #[test]
    fn prepared_blur_matches_small_geometric_halo_after_extreme_uniform_scales() {
        use crate::filter::{FilterData, PreparedFilter};
        use crate::filter_effects::{EdgeMode, Filter, FilterPrimitive};
        use crate::kurbo::Affine;
        for (sigma, scale, expected) in [
            (1.0e11, 1.0e-12, 0.1),
            (1.0e-20, 1.0e20, 1.0),
            (1.0, 0.0, 0.0),
            (f32::MAX, 1.0e-38, (f64::from(f32::MAX) * 1.0e-38) as f32),
        ] {
            let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: sigma,
                edge_mode: EdgeMode::None,
            });
            let data = FilterData::new(filter, Affine::scale(scale));
            let PreparedFilter::GaussianBlur(blur) = data.prepare_for_layer() else {
                panic!("expected gaussian blur")
            };
            assert!(
                (blur.std_deviation - expected).abs() <= expected * 1.0e-6,
                "sigma {sigma} scale {scale}: {} != {expected}",
                blur.std_deviation
            );
            assert!(
                blur.n_decimations <= 1,
                "small physical blur must not encode deep decimation"
            );
            assert!(data.source_padding.left as f32 >= 3.0 * blur.std_deviation);
            assert!(data.source_padding.top as f32 >= 3.0 * blur.std_deviation);
        }
    }

    #[test]
    fn blur_scale_preserves_zero_and_singular_values_without_flooring() {
        use crate::kurbo::Affine;
        for (transform, sigma, expected) in [
            (Affine::scale(0.0), 1.0e11, 0.0),
            (Affine::scale_non_uniform(0.0, 2.0), 1.0, 1.0),
            (Affine::scale_non_uniform(1.0e20, 1.0), 1.0e-20, 0.5),
            (
                Affine::new([1.0e20, 0.0, 1.0e20, 1.0e20, 0.0, 0.0]),
                1.0e-20,
                5.0_f32.sqrt() / 2.0,
            ),
        ] {
            let actual = super::transform_blur_params(sigma, &transform);
            assert!(
                (actual - expected).abs() <= expected * 1.0e-7,
                "{transform:?}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn ordinary_blur_scales_retain_exact_existing_rounding() {
        use crate::kurbo::Affine;
        for transform in [
            Affine::IDENTITY,
            Affine::scale(0.5),
            Affine::scale_non_uniform(2.0, 3.0),
            Affine::rotate(0.75),
            Affine::new([1.0, 0.25, 0.75, 2.0, 3.0, 4.0]),
        ] {
            let (x, y) = crate::util::extract_scales(&transform);
            for sigma in [0.0, 0.25, 1.0, 8.0, 150.0] {
                assert_eq!(
                    super::transform_blur_params(sigma, &transform).to_bits(),
                    (sigma * ((x + y) / 2.0)).to_bits()
                );
            }
        }
    }
}
