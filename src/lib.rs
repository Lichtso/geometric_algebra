#![cfg_attr(all(target_arch = "wasm32", target_feature = "simd128"), feature(wasm_simd))]
#![cfg_attr(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"), feature(stdsimd))]

pub mod simd;
pub mod complex;
pub mod ppga2d;
pub mod ppga3d;

impl complex::Scalar {
    pub const fn new(real: f32) -> Self {
        Self { g0: real }
    }

    pub fn real(self) -> f32 {
        self.g0
    }

    pub fn sqrt(self) -> complex::MultiVector {
        if self.g0 < 0.0 {
            complex::MultiVector::new(0.0, (-self.g0).sqrt())
        } else {
            complex::MultiVector::new(self.g0.sqrt(), 0.0)
        }
    }
}

impl complex::MultiVector {
    pub const fn new(real: f32, imaginary: f32) -> Self {
        Self {
            g0: simd::Simd32x2 {
                f32x2: [real, imaginary],
            },
        }
    }

    pub fn real(self) -> f32 {
        self.g0.get_f(0)
    }

    pub fn imaginary(self) -> f32 {
        self.g0.get_f(1)
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
pub trait Automorph {
    type Output;
    fn automorph(self) -> Self::Output;
}

/// Negates elements with `grade % 4 >= 2`
///
/// Also called reversion
pub trait Transpose {
    type Output;
    fn transpose(self) -> Self::Output;
}

/// Negates elements with `(grade + 3) % 4 < 2`
pub trait Conjugate {
    type Output;
    fn conjugate(self) -> Self::Output;
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

/// `self * other * self`
/// 
/// Basically a sandwich product without an involution
pub trait Reflection<T> {
    type Output;
    fn reflection(self, other: T) -> Self::Output;
}

/// `self * other * self.transpose()`
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

/// Direction without magnitude (set to scalar `1.0`)
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
