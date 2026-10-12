// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What the filter of a filter layer does: how far its source viewport and
//! the pixmap it runs on reach, by how many whole pixels it moves its source,
//! and the blur it applies.
//!
//! The paddings are those of [`FilterData`] and the offsets those of
//! [`PreparedFilter`], which the recorder uses, wherever the `f64` and `f32`
//! values they come from are finite and they do not saturate. Past that, they
//! are computed exactly: offsets and the coefficients of the transform are
//! binary fractions, and so are their products and sums, so nothing saturates
//! before it is counted in whole pixels, and opposite moves of nested filters
//! cancel.

use crate::filter::placed::{Coord, Padding, Position};
use crate::filter::pyramid::Blur;
use crate::kurbo::{Affine, Rect};
use core::cmp::Ordering;
use num_bigint::{BigInt, Sign};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::filter_effects::{EdgeMode, FilterPrimitive};
use vello_common::geometry::PaddingU64;
use vello_common::tile::Tile;

/// The binary fraction `mantissa * 2^exponent`.
#[derive(Clone, Debug)]
struct Dyadic {
    mantissa: BigInt,
    exponent: i64,
}

impl Dyadic {
    const ZERO: Self = Self {
        mantissa: BigInt::ZERO,
        exponent: 0,
    };

    /// Return the value of `value`, which is finite.
    fn of(value: f64) -> Self {
        debug_assert!(value.is_finite(), "only finite values are binary fractions");
        let bits = value.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1 << 52) - 1);
        let (mantissa, exponent) = if biased == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), biased - 1075)
        };
        let mantissa = BigInt::from(mantissa);
        Self {
            mantissa: if value.is_sign_negative() {
                -mantissa
            } else {
                mantissa
            },
            exponent,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            mantissa: &self.mantissa * &other.mantissa,
            exponent: self.exponent + other.exponent,
        }
    }

    fn add(&self, other: &Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        let aligned = |value: &Self| &value.mantissa << (value.exponent - exponent).unsigned_abs();
        Self {
            mantissa: aligned(self) + aligned(other),
            exponent,
        }
    }

    fn neg(&self) -> Self {
        Self {
            mantissa: -&self.mantissa,
            exponent: self.exponent,
        }
    }

    fn cmp(&self, other: &Self) -> Ordering {
        match self.add(&other.neg()).mantissa.sign() {
            Sign::Minus => Ordering::Less,
            Sign::NoSign => Ordering::Equal,
            Sign::Plus => Ordering::Greater,
        }
    }

    fn min(self, other: Self) -> Self {
        if self.cmp(&other) == Ordering::Greater {
            other
        } else {
            self
        }
    }

    fn max(self, other: Self) -> Self {
        if self.cmp(&other) == Ordering::Less {
            other
        } else {
            self
        }
    }

    /// Return the smallest multiple of `2^levels` at least the value.
    fn ceil_to(&self, levels: u32) -> Coord {
        let levels = i64::from(levels);
        if self.exponent >= levels {
            return Coord::new(&self.mantissa << self.exponent.unsigned_abs());
        }
        // Shifting right rounds towards negative infinity.
        let steps = -(-&self.mantissa >> (levels - self.exponent).unsigned_abs());
        Coord::new(steps << levels.unsigned_abs())
    }

    /// Return the nearest whole number, halves rounded away from zero.
    fn round(&self) -> Coord {
        if self.exponent >= 0 {
            return Coord::new(&self.mantissa << self.exponent.unsigned_abs());
        }
        let shift = self.exponent.unsigned_abs();
        let half = BigInt::from(1) << (shift - 1);
        let magnitude = (self.mantissa.magnitude().clone() + half.magnitude()) >> shift;
        Coord::new(BigInt::from_biguint(self.mantissa.sign(), magnitude))
    }
}

/// A rectangle of binary fractions around the origin.
#[derive(Clone, Debug)]
struct Expansion {
    x0: Dyadic,
    y0: Dyadic,
    x1: Dyadic,
    y1: Dyadic,
}

impl Expansion {
    /// Return the expansion covering the origin and `(x0, y0)..(x1, y1)`.
    fn new(x0: Dyadic, y0: Dyadic, x1: Dyadic, y1: Dyadic) -> Self {
        Self {
            x0: x0.min(Dyadic::ZERO),
            y0: y0.min(Dyadic::ZERO),
            x1: x1.max(Dyadic::ZERO),
            y1: y1.max(Dyadic::ZERO),
        }
    }

    /// Return the expansion of a blur of `radius` around a copy moved by `offset`.
    fn around((dx, dy): &(Dyadic, Dyadic), radius: &Dyadic) -> Self {
        Self::new(
            dx.add(&radius.neg()),
            dy.add(&radius.neg()),
            dx.add(radius),
            dy.add(radius),
        )
    }

    fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    /// Return the bounding box of the expansion mapped by `linear`.
    fn transformed(&self, linear: &[Dyadic; 4]) -> Self {
        let [a, b, c, d] = linear;
        let corners = [
            (&self.x0, &self.y0),
            (&self.x1, &self.y0),
            (&self.x0, &self.y1),
            (&self.x1, &self.y1),
        ]
        .map(|(x, y)| (a.mul(x).add(&c.mul(y)), b.mul(x).add(&d.mul(y))));
        let [first, rest @ ..] = corners;
        rest.into_iter().fold(
            Self {
                x0: first.0.clone(),
                y0: first.1.clone(),
                x1: first.0,
                y1: first.1,
            },
            |bounds, (x, y)| Self {
                x0: bounds.x0.min(x.clone()),
                y0: bounds.y0.min(y.clone()),
                x1: bounds.x1.max(x),
                y1: bounds.y1.max(y),
            },
        )
    }

    /// Return the padding covering the expansion in whole tiles.
    fn padding(&self) -> Padding {
        let (width, height) = (Tile::WIDTH.trailing_zeros(), Tile::HEIGHT.trailing_zeros());
        Padding {
            left: self.x0.neg().ceil_to(width),
            top: self.y0.neg().ceil_to(height),
            right: self.x1.ceil_to(width),
            bottom: self.y1.ceil_to(height),
        }
    }
}

/// What a filter layer's filter does.
#[derive(Clone, Debug)]
pub(crate) struct Reach {
    /// How far the viewport of the layer's source reaches past the viewport
    /// of its parent.
    pub(crate) source_padding: Padding,
    /// How far the pixmap the filter runs on reaches past the content of the source.
    pub(crate) filter_padding: Padding,
    pub(crate) effect: Effect,
}

/// How the output of a filter is computed from its source on the pixmap it runs on.
#[derive(Clone, Debug)]
pub(crate) enum Effect {
    /// The prepared flood fills the pixmap.
    Flood,
    /// The source moved by whole pixels; what leaves the pixmap is lost.
    Offset(Position),
    /// The source blurred, or the source itself without a blur.
    Blur(Option<Blur>),
    /// The source moved as by an offset, blurred unless there is no blur, and
    /// recolored, under the source itself if `keeps_source` is set.
    Shadow {
        offset: Position,
        blur: Option<Blur>,
        color: AlphaColor<Srgb>,
        keeps_source: bool,
    },
}

impl Reach {
    /// Return whether the output depends on where the pixmap ends, beyond
    /// covering the content of the source and the padding.
    pub(crate) fn reads_extent(&self) -> bool {
        match &self.effect {
            Effect::Flood => true,
            Effect::Offset(_) => false,
            Effect::Blur(blur) | Effect::Shadow { blur, .. } => blur.is_some(),
        }
    }

    /// Return what `filter`, which has a single primitive, does.
    pub(crate) fn new(filter: &FilterData) -> Self {
        let prepared = PreparedFilter::new(&filter.filter, &filter.transform);
        let [a, b, c, d, _, _] = filter.transform.as_coeffs();
        let linear = [a, b, c, d].map(Dyadic::of);
        let back = |(dx, dy): &(Dyadic, Dyadic)| (dx.neg(), dy.neg());
        let zero = || Expansion::new(Dyadic::ZERO, Dyadic::ZERO, Dyadic::ZERO, Dyadic::ZERO);
        // The expansions in user space, and those of the move and the
        // isotropic blur applied in device space.
        let (filter_expansion, source_expansion, device) = match &filter.filter.graph.primitives[0]
        {
            FilterPrimitive::GaussianBlur { std_deviation, .. } => {
                let user = radius(*std_deviation);
                let around = Expansion::around(&(Dyadic::ZERO, Dyadic::ZERO), &user);
                let PreparedFilter::GaussianBlur(blur) = &prepared else {
                    unreachable!("a blur prepares as a blur");
                };
                let device = radius(blur.std_deviation);
                let device = Expansion::around(&(Dyadic::ZERO, Dyadic::ZERO), &device);
                (around.clone(), around, Some((device.clone(), device)))
            }
            FilterPrimitive::Offset { dx, dy } => {
                let offset = (Dyadic::of((*dx).into()), Dyadic::of((*dy).into()));
                let device_offset = device_offset(&filter.transform, *dx, *dy);
                let none = Dyadic::ZERO;
                (
                    Expansion::around(&offset, &none),
                    Expansion::around(&back(&offset), &none),
                    Some((
                        Expansion::around(&device_offset, &none),
                        Expansion::around(&back(&device_offset), &none),
                    )),
                )
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                ..
            }
            | FilterPrimitive::DropShadowOnly {
                dx,
                dy,
                std_deviation,
                ..
            } => {
                let offset = (Dyadic::of((*dx).into()), Dyadic::of((*dy).into()));
                let user = radius(*std_deviation);
                let PreparedFilter::DropShadow(shadow) = &prepared else {
                    unreachable!("a drop shadow prepares as a drop shadow");
                };
                let device_offset = device_offset(&filter.transform, *dx, *dy);
                let device = radius(shadow.std_deviation);
                (
                    Expansion::around(&offset, &user),
                    Expansion::around(&back(&offset), &user),
                    Some((
                        Expansion::around(&device_offset, &device),
                        Expansion::around(&back(&device_offset), &device),
                    )),
                )
            }
            _ => (zero(), zero(), None),
        };
        let mut filter_expansion = filter_expansion.transformed(&linear);
        let mut source_expansion = source_expansion.transformed(&linear);
        if let Some((filter_device, source_device)) = device {
            filter_expansion = filter_expansion.union(filter_device);
            source_expansion = source_expansion.union(source_device);
        }
        let effect = match prepared {
            PreparedFilter::Flood(_) => Effect::Flood,
            PreparedFilter::Offset(_) => {
                let FilterPrimitive::Offset { dx, dy } = &filter.filter.graph.primitives[0] else {
                    unreachable!("an offset prepares from an offset");
                };
                Effect::Offset(whole_pixels(&filter.transform, *dx, *dy))
            }
            PreparedFilter::GaussianBlur(blur) => {
                Effect::Blur((blur.std_deviation > 0.0).then(|| Blur {
                    decimations: u32::try_from(blur.n_decimations).unwrap(),
                    kernel: blur.kernel[..usize::from(blur.kernel_size)].to_vec(),
                    edge_mode: blur.edge_mode,
                }))
            }
            PreparedFilter::DropShadow(shadow) => {
                let (FilterPrimitive::DropShadow { dx, dy, .. }
                | FilterPrimitive::DropShadowOnly { dx, dy, .. }) =
                    &filter.filter.graph.primitives[0]
                else {
                    unreachable!("a drop shadow prepares from a drop shadow");
                };
                Effect::Shadow {
                    offset: whole_pixels(&filter.transform, *dx, *dy),
                    blur: (shadow.std_deviation > 0.0).then(|| Blur {
                        decimations: u32::try_from(shadow.n_decimations).unwrap(),
                        kernel: shadow.kernel[..usize::from(shadow.kernel_size)].to_vec(),
                        edge_mode: shadow.edge_mode,
                    }),
                    color: shadow.color,
                    keeps_source: shadow.composite_original,
                }
            }
        };
        let (filter_padding, source_padding) = recorded_paddings(filter)
            .unwrap_or_else(|| (filter_expansion.padding(), source_expansion.padding()));
        Self {
            source_padding,
            filter_padding,
            effect,
        }
    }
}

/// Return the filter and source paddings the recorder places `filter` with,
/// unless they leave the `u16` domain it places pixmaps in, or a corner of the
/// expansions they cover overflows `f64` under the transform, which the
/// bounding box of the corners would skip.
fn recorded_paddings(filter: &FilterData) -> Option<(Padding, Padding)> {
    let [a, b, c, d, _, _] = filter.transform.as_coeffs();
    let finite = |rect: Rect| {
        [rect.x0, rect.x1].into_iter().all(|x| {
            [rect.y0, rect.y1]
                .into_iter()
                .all(|y| (a * x + c * y).is_finite() && (b * x + d * y).is_finite())
        })
    };
    let graph = &filter.filter.graph;
    if !finite(graph.filter_expansion(&Affine::IDENTITY))
        || !finite(graph.source_expansion(&Affine::IDENTITY))
    {
        return None;
    }
    let padding = |padding: PaddingU64| {
        let sides = [padding.left, padding.top, padding.right, padding.bottom];
        let fits = sides.iter().all(|&side| side <= u64::from(u16::MAX));
        fits.then(|| Padding {
            left: padding.left.into(),
            top: padding.top.into(),
            right: padding.right.into(),
            bottom: padding.bottom.into(),
        })
    };
    Some((padding(filter.filter_padding)?, padding(filter.source_padding)?))
}

/// Return an offset by `(dx, dy)` under the linear part of `transform` in
/// device space: transformed in `f64` and converted to `f32`, as
/// [`PreparedFilter`] computes it, or where `f32` overflows, exactly.
fn device_offset(transform: &Affine, dx: f32, dy: f32) -> (Dyadic, Dyadic) {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let (dx, dy) = (f64::from(dx), f64::from(dy));
    let axis = |x: f64, y: f64| {
        let value = (x * dx + y * dy) as f32;
        if value.is_finite() {
            Dyadic::of(value.into())
        } else {
            let exact = |factor: f64, offset: f64| Dyadic::of(factor).mul(&Dyadic::of(offset));
            exact(x, dx).add(&exact(y, dy))
        }
    };
    (axis(a, c), axis(b, d))
}

/// Return the whole pixels an offset by `(dx, dy)` moves by under the linear
/// part of `transform`: its device offset rounded, as the filter rounds it.
fn whole_pixels(transform: &Affine, dx: f32, dy: f32) -> Position {
    let (dx, dy) = device_offset(transform, dx, dy);
    (dx.round(), dy.round())
}

/// Return the radius a blur of `std_deviation` reaches, as filter expansions count it.
fn radius(std_deviation: f32) -> Dyadic {
    let blur = FilterPrimitive::GaussianBlur {
        std_deviation,
        edge_mode: EdgeMode::None,
    };
    Dyadic::of(blur.filter_expansion().x1)
}

#[cfg(test)]
mod tests {
    use super::{Dyadic, Effect, Reach};
    use crate::filter::placed::{Coord, Padding};
    use crate::kurbo::Affine;
    use vello_common::color::palette::css::RED;
    use vello_common::filter::FilterData;
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::geometry::PaddingU64;

    fn reach(primitive: FilterPrimitive, transform: Affine) -> Reach {
        Reach::new(&FilterData::new(
            Filter::from_primitive(primitive),
            transform,
        ))
    }

    #[test]
    fn binary_fractions_round_and_snap_exactly() {
        let of = Dyadic::of;
        assert_eq!(of(2.5).round(), Coord::from(3_i64));
        assert_eq!(of(-2.5).round(), Coord::from(-3_i64));
        assert_eq!(of(-2.25).round(), Coord::from(-2_i64));
        assert_eq!(of(0.49999999999999994).round(), Coord::ZERO);
        assert_eq!(of(4.0).ceil_to(2), Coord::from(4_i64));
        assert_eq!(of(4.000001).ceil_to(2), Coord::from(8_i64));
        assert_eq!(of(-3.5).ceil_to(2), Coord::ZERO);
        assert_eq!(of(f64::MIN_POSITIVE).ceil_to(2), Coord::from(4_i64));
        let max = of(f64::from(f32::MAX));
        assert_eq!(max.mul(&of(2.0)).add(&max.neg()).round(), max.round());
    }

    #[test]
    fn offsets_scaled_past_f32_cancel_exactly() {
        let offset =
            |dx: f32, transform| match reach(FilterPrimitive::Offset { dx, dy: 0.0 }, transform)
                .effect
            {
                Effect::Offset(offset) => offset,
                _ => unreachable!(),
            };
        let inner = offset(f32::MAX, Affine::scale(2.0));
        let outer = offset(-f32::MAX, Affine::IDENTITY);
        assert_eq!(&inner.0 + &outer.0, offset(f32::MAX, Affine::IDENTITY).0);
        assert_eq!(&inner.1 + &outer.1, Coord::ZERO);
        assert_eq!(
            &offset(f32::MAX, Affine::IDENTITY).0 + &outer.0,
            Coord::ZERO
        );
    }

    #[test]
    fn offsets_round_through_f32_as_the_filter_does() {
        // 0.5 scaled by 0.99999999 is 0.499999995, which `f32` holds as 0.5.
        let transform = Affine::scale_non_uniform(0.999_999_99, 1.0);
        for dx in [0.5, -0.5] {
            let offset = match reach(FilterPrimitive::Offset { dx, dy: dx }, transform).effect {
                Effect::Offset(offset) => offset,
                _ => unreachable!(),
            };
            let shadow = FilterPrimitive::DropShadow {
                dx,
                dy: dx,
                std_deviation: 1.0,
                color: RED,
                edge_mode: EdgeMode::None,
            };
            let shadow = match reach(shadow, transform).effect {
                Effect::Shadow { offset, .. } => offset,
                _ => unreachable!(),
            };
            let moved = Coord::from(if dx < 0.0 { -1_i64 } else { 1 });
            assert_eq!(offset, (moved.clone(), moved.clone()), "offset by {dx}");
            assert_eq!(shadow, (moved.clone(), moved), "shadow by {dx}");
        }
    }

    #[test]
    fn paddings_are_those_the_recorder_computes() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let scale = [0.1, 0.2, 0.3, 0.7, 1.1, 3.0][(next() % 6) as usize];
            let angle = (next() % 1000) as f64 / 1000.0;
            let transform = Affine::rotate(angle) * Affine::scale(scale);
            let value = (next() % 4000) as f32 / 30.0;
            let offset = (next() % 4000) as f32 / 7.0 - 200.0;
            let primitive = match next() % 3 {
                0 => FilterPrimitive::GaussianBlur {
                    std_deviation: value,
                    edge_mode: EdgeMode::None,
                },
                1 => FilterPrimitive::Offset {
                    dx: offset,
                    dy: value,
                },
                _ => FilterPrimitive::DropShadow {
                    dx: offset,
                    dy: -value,
                    std_deviation: value / 3.0,
                    color: RED,
                    edge_mode: EdgeMode::None,
                },
            };
            let data = FilterData::new(Filter::from_primitive(primitive.clone()), transform);
            let reach = Reach::new(&data);
            let padding = |padding: PaddingU64| Padding {
                left: padding.left.into(),
                top: padding.top.into(),
                right: padding.right.into(),
                bottom: padding.bottom.into(),
            };
            assert_eq!(
                (reach.filter_padding, reach.source_padding),
                (padding(data.filter_padding), padding(data.source_padding)),
                "{primitive:?} under {transform:?}"
            );
        }
    }

    #[test]
    fn paddings_cover_offsets_and_blurs_exactly() {
        let shadow = reach(
            FilterPrimitive::DropShadow {
                dx: 10.0,
                dy: -6.0,
                std_deviation: 2.0,
                color: RED,
                edge_mode: EdgeMode::None,
            },
            Affine::IDENTITY,
        );
        let padding = |left: i64, top: i64, right: i64, bottom: i64| Padding {
            left: left.into(),
            top: top.into(),
            right: right.into(),
            bottom: bottom.into(),
        };
        // The shadow reaches 6 around the moved copy, which lies right and above.
        assert_eq!(shadow.filter_padding, padding(0, 12, 16, 0));
        assert_eq!(shadow.source_padding, padding(16, 0, 0, 12));
        let far = reach(
            FilterPrimitive::Offset {
                dx: f32::MAX,
                dy: 0.0,
            },
            Affine::scale(4.0),
        );
        let moved = Coord::floor_of(f64::from(f32::MAX)).shl(2);
        assert_eq!(
            far.filter_padding,
            Padding {
                right: moved.clone(),
                ..padding(0, 0, 0, 0)
            }
        );
        assert_eq!(
            far.source_padding,
            Padding {
                left: moved,
                ..padding(0, 0, 0, 0)
            }
        );
    }
}

