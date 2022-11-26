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
    pub fn real(self) -> f32 {
        self[0]
    }

    pub fn sqrt(self) -> epga1d::ComplexNumber {
        if self[0] < 0.0 {
            epga1d::ComplexNumber::from([0.0, (-self[0]).sqrt()])
        } else {
            epga1d::ComplexNumber::from([self[0].sqrt(), 0.0])
        }
    }
}

impl epga1d::ComplexNumber {
    pub fn real(self) -> f32 {
        self[0]
    }

    pub fn imaginary(self) -> f32 {
        self[1]
    }

    pub fn from_polar(magnitude: f32, argument: f32) -> Self {
        Self::from([magnitude * argument.cos(), magnitude * argument.sin()])
    }

    pub fn arg(self) -> f32 {
        self.imaginary().atan2(self.real())
    }
}

impl Exp for epga1d::ComplexNumber {
    type Output = Self;

    fn exp(self) -> Self {
        Self::from_polar(self[0].exp(), self[1])
    }
}

impl Ln for epga1d::ComplexNumber {
    type Output = Self;

    fn ln(self) -> Self {
        Self::from([self.magnitude()[0].ln(), self.arg()])
    }
}

impl Powf for epga1d::ComplexNumber {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        Self::from_polar(self.magnitude()[0].powf(exponent), self.arg() * exponent)
    }
}

impl Exp for ppga2d::IdealPoint {
    type Output = ppga2d::Translator;

    fn exp(self) -> ppga2d::Translator {
        ppga2d::Translator::from([1.0, self[0], self[1]])
    }
}

impl Ln for ppga2d::Translator {
    type Output = ppga2d::IdealPoint;

    fn ln(self) -> ppga2d::IdealPoint {
        let result: ppga2d::IdealPoint = self.into();
        result / ppga2d::Scalar::from([self[0]])
    }
}

impl Powf for ppga2d::Translator {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        self.ln()
            .geometric_product(ppga2d::Scalar::from([exponent]))
            .exp()
    }
}

impl Exp for ppga2d::Point {
    type Output = ppga2d::Motor;

    fn exp(self) -> ppga2d::Motor {
        let det = self[0] * self[0];
        if det <= 0.0 {
            return ppga2d::Motor::from([1.0, 0.0, self[1], self[2]]);
        }
        let a = det.sqrt();
        let c = a.cos();
        let s = a.sin() / a;
        let g0 = simd::Simd32x3::from(s) * self.group0();
        ppga2d::Motor::from([c, g0[0], g0[1], g0[2]])
    }
}

impl Ln for ppga2d::Motor {
    type Output = ppga2d::Point;

    fn ln(self) -> ppga2d::Point {
        let det = 1.0 - self[0] * self[0];
        if det <= 0.0 {
            return ppga2d::Point::from([0.0, self[2], self[3]]);
        }
        let a = 1.0 / det;
        let b = self[0].acos() * a.sqrt();
        let g0 = simd::Simd32x4::from(b) * self.group0();
        return ppga2d::Point::from([g0[1], g0[2], g0[3]]);
    }
}

impl Powf for ppga2d::Motor {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        self.ln()
            .geometric_product(ppga2d::Scalar::from([exponent]))
            .exp()
    }
}

impl Exp for ppga3d::IdealPoint {
    type Output = ppga3d::Translator;

    fn exp(self) -> ppga3d::Translator {
        ppga3d::Translator::from([1.0, self[0], self[1], self[2]])
    }
}

impl Ln for ppga3d::Translator {
    type Output = ppga3d::IdealPoint;

    fn ln(self) -> ppga3d::IdealPoint {
        let result: ppga3d::IdealPoint = self.into();
        result / ppga3d::Scalar::from([self[0]])
    }
}

impl Powf for ppga3d::Translator {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        self.ln()
            .geometric_product(ppga3d::Scalar::from([exponent]))
            .exp()
    }
}

impl Exp for ppga3d::Line {
    type Output = ppga3d::Motor;

    fn exp(self) -> ppga3d::Motor {
        let det = self[3] * self[3] + self[4] * self[4] + self[5] * self[5];
        if det <= 0.0 {
            return ppga3d::Motor::from([1.0, 0.0, 0.0, 0.0, 0.0, self[0], self[1], self[2]]);
        }
        let a = det.sqrt();
        let c = a.cos();
        let s = a.sin() / a;
        let m = self[0] * self[3] + self[1] * self[4] + self[2] * self[5];
        let t = m / det * (c - s);
        let g0 = simd::Simd32x3::from(s) * self.group1();
        let g1 = simd::Simd32x3::from(s) * self.group0() + simd::Simd32x3::from(t) * self.group1();
        ppga3d::Motor::from([c, g0[0], g0[1], g0[2], s * m, g1[0], g1[1], g1[2]])
    }
}

impl Ln for ppga3d::Motor {
    type Output = ppga3d::Line;

    fn ln(self) -> ppga3d::Line {
        let det = 1.0 - self[0] * self[0];
        if det <= 0.0 {
            return ppga3d::Line::from([self[5], self[6], self[7], 0.0, 0.0, 0.0]);
        }
        let a = 1.0 / det;
        let b = self[0].acos() * a.sqrt();
        let c = a * self[4] * (1.0 - self[0] * b);
        let g0 = simd::Simd32x4::from(b) * self.group1() + simd::Simd32x4::from(c) * self.group0();
        let g1 = simd::Simd32x4::from(b) * self.group0();
        return ppga3d::Line::from([g0[1], g0[2], g0[3], g1[1], g1[2], g1[3]]);
    }
}

impl Powf for ppga3d::Motor {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        self.ln()
            .geometric_product(ppga3d::Scalar::from([exponent]))
            .exp()
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

/// Raises a number to the scalar power of `-1.0`
pub trait Inverse {
    type Output;
    fn inverse(self) -> Self::Output;
}

/// The natural logarithm
pub trait Ln {
    type Output;
    fn ln(self) -> Self::Output;
}

/// The exponential function
pub trait Exp {
    type Output;
    fn exp(self) -> Self::Output;
}

/// Raises a number to an integer scalar power
pub trait Powi {
    type Output;
    fn powi(self, exponent: isize) -> Self::Output;
}

/// Raises a number to an floating point scalar power
pub trait Powf {
    type Output;
    fn powf(self, exponent: f32) -> Self::Output;
}
