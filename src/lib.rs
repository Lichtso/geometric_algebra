#![cfg_attr(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"), feature(stdsimd))]

pub mod epga1d;
pub mod ppga1d;
pub mod hpga1d;
pub mod epga2d;
pub mod ppga2d;
pub mod hpga2d;
pub mod epga3d;
pub mod ppga3d;
pub mod hpga3d;
pub mod simd;

impl epga1d::Scalar {
    pub const fn new(real: f32) -> Self {
        Self { g0: real }
    }

    pub fn real(self) -> f32 {
        self.g0
    }

    pub fn sqrt(self) -> epga1d::ComplexNumber {
        if self.g0 < 0.0 {
            epga1d::ComplexNumber::new(0.0, (-self.g0).sqrt())
        } else {
            epga1d::ComplexNumber::new(self.g0.sqrt(), 0.0)
        }
    }
}

impl epga1d::ComplexNumber {
    pub const fn new(real: f32, imaginary: f32) -> Self {
        Self {
            g0: simd::Simd32x2 {
                f32x2: [real, imaginary],
            },
        }
    }

    pub fn real(self) -> f32 {
        self.g0[0]
    }

    pub fn imaginary(self) -> f32 {
        self.g0[1]
    }

    pub fn from_polar(magnitude: f32, angle: f32) -> Self {
        Self::new(magnitude * angle.cos(), magnitude * angle.sin())
    }

    pub fn arg(self) -> f32 {
        self.imaginary().atan2(self.real())
    }

    pub fn powf(self, exponent: f32) -> Self {
        Self::from_polar(self.magnitude().g0.powf(exponent), self.arg() * exponent)
    }
}

/// All elements set to `0.0`
pub trait Zero {
    fn zero() -> Self;
}

/// All elements set to `0.0`, except for the scalar, which is set to `1.0`
pub trait One {
    fn one() -> Self;
}

/// Element order reversed
pub trait Dual {
    type Output;
    fn dual(self) -> Self::Output;
}

/// Negates elements with `grade % 2 == 1`
///
/// Also called main involution
pub trait Automorphism {
    type Output;
    fn automorphism(self) -> Self::Output;
}

/// Negates elements with `grade % 4 >= 2`
///
/// Also called transpose
pub trait Reversal {
    type Output;
    fn reversal(self) -> Self::Output;
}

/// Negates elements with `(grade + 3) % 4 < 2`
pub trait Conjugation {
    type Output;
    fn conjugation(self) -> Self::Output;
}

/// General multi vector multiplication
pub trait GeometricProduct<T> {
    type Output;
    fn geometric_product(self, other: T) -> Self::Output;
}

/// Dual of the geometric product grade filtered by `t == r + s`
///
/// Also called join
pub trait RegressiveProduct<T> {
    type Output;
    fn regressive_product(self, other: T) -> Self::Output;
}

/// Geometric product grade filtered by `t == r + s`
///
/// Also called meet or exterior product
pub trait OuterProduct<T> {
    type Output;
    fn outer_product(self, other: T) -> Self::Output;
}

/// Geometric product grade filtered by `t == (r - s).abs()`
///
/// Also called fat dot product
pub trait InnerProduct<T> {
    type Output;
    fn inner_product(self, other: T) -> Self::Output;
}

/// Geometric product grade filtered by `t == s - r`
pub trait LeftContraction<T> {
    type Output;
    fn left_contraction(self, other: T) -> Self::Output;
}

/// Geometric product grade filtered by `t == r - s`
pub trait RightContraction<T> {
    type Output;
    fn right_contraction(self, other: T) -> Self::Output;
}

/// Geometric product grade filtered by `t == 0`
pub trait ScalarProduct<T> {
    type Output;
    fn scalar_product(self, other: T) -> Self::Output;
}

/// `self * other * self.reversion()`
///
/// Also called sandwich product
pub trait Transformation<T> {
    type Output;
    fn transformation(self, other: T) -> Self::Output;
}

/// Square of the magnitude
pub trait SquaredMagnitude {
    type Output;
    fn squared_magnitude(self) -> Self::Output;
}

/// Length as scalar
///
/// Also called amplitude, absolute value or norm
pub trait Magnitude {
    type Output;
    fn magnitude(self) -> Self::Output;
}

/// Direction without magnitude (set to scalar `-1.0` or `1.0`)
///
/// Also called sign or normalize
pub trait Signum {
    type Output;
    fn signum(self) -> Self::Output;
}

/// Exponentiation by scalar `-1.0`
pub trait Inverse {
    type Output;
    fn inverse(self) -> Self::Output;
}

/// Exponentiation by a scalar integer
pub trait Powi {
    type Output;
    fn powi(self, exponent: isize) -> Self::Output;
}
