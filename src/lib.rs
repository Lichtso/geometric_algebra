#![cfg_attr(all(target_arch = "wasm32", target_feature = "simd128"), feature(wasm_simd))]
#![cfg_attr(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"), feature(stdsimd))]

pub use simd::*;
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

impl ppga2d::Rotor {
    pub fn from_angle(mut angle: f32) -> Self {
        angle *= 0.5;
        Self {
            g0: simd::Simd32x2::from([angle.cos(), angle.sin()]),
        }
    }

    pub fn angle(self) -> f32 {
        self.g0.get_f(1).atan2(self.g0.get_f(0)) * 2.0
    }
}

impl ppga2d::Point {
    pub fn from_coordinates(coordinates: [f32; 2]) -> Self {
        Self {
            g0: simd::Simd32x3::from([1.0, coordinates[0], coordinates[1]]),
        }
    }

    pub fn from_direction(coordinates: [f32; 2]) -> Self {
        Self {
            g0: simd::Simd32x3::from([0.0, coordinates[0], coordinates[1]]),
        }
    }
}

impl ppga2d::Plane {
    pub fn from_normal_and_distance(normal: [f32; 2], distance: f32) -> Self {
        Self {
            g0: simd::Simd32x3::from([distance, normal[1], -normal[0]]),
        }
    }
}

impl ppga2d::Translator {
    pub fn from_coordinates(coordinates: [f32; 2]) -> Self {
        Self {
            g0: simd::Simd32x3::from([1.0, coordinates[1] * 0.5, coordinates[0] * -0.5]),
        }
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

/// Also called reversion
pub trait Transpose {
    type Output;
    fn transpose(self) -> Self::Output;
}

/// Also called involution
pub trait Automorph {
    type Output;
    fn automorph(self) -> Self::Output;
}

pub trait Conjugate {
    type Output;
    fn conjugate(self) -> Self::Output;
}

pub trait GeometricProduct<T> {
    type Output;
    fn geometric_product(self, other: T) -> Self::Output;
}

/// Also called join
pub trait RegressiveProduct<T> {
    type Output;
    fn regressive_product(self, other: T) -> Self::Output;
}

/// Also called meet or exterior product
pub trait OuterProduct<T> {
    type Output;
    fn outer_product(self, other: T) -> Self::Output;
}

/// Also called fat dot product
pub trait InnerProduct<T> {
    type Output;
    fn inner_product(self, other: T) -> Self::Output;
}

pub trait LeftContraction<T> {
    type Output;
    fn left_contraction(self, other: T) -> Self::Output;
}

pub trait RightContraction<T> {
    type Output;
    fn right_contraction(self, other: T) -> Self::Output;
}

pub trait ScalarProduct<T> {
    type Output;
    fn scalar_product(self, other: T) -> Self::Output;
}

pub trait Reflection<T> {
    type Output;
    fn reflection(self, other: T) -> Self::Output;
}

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

/// Also called amplitude, absolute value or norm
pub trait Magnitude {
    type Output;
    fn magnitude(self) -> Self::Output;
}

/// Also called normalize
pub trait Signum {
    type Output;
    fn signum(self) -> Self::Output;
}

/// Exponentiation by scalar negative one
pub trait Inverse {
    type Output;
    fn inverse(self) -> Self::Output;
}

/// Exponentiation by a scalar integer
pub trait Powi {
    type Output;
    fn powi(self, exponent: isize) -> Self::Output;
}
