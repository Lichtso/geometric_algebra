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

    pub fn from_polar(magnitude: f32, argument: f32) -> Self {
        Self::new(magnitude * argument.cos(), magnitude * argument.sin())
    }

    pub fn arg(self) -> f32 {
        self.imaginary().atan2(self.real())
    }
}

impl Exp for epga1d::ComplexNumber {
    type Output = Self;

    fn exp(self) -> Self {
        Self::from_polar(self.g0[0].exp(), self.g0[1])
    }
}

impl Ln for epga1d::ComplexNumber {
    type Output = Self;

    fn ln(self) -> Self {
        Self::new(self.magnitude().g0.ln(), self.arg())
    }
}

impl Powf for epga1d::ComplexNumber {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        Self::from_polar(self.magnitude().g0.powf(exponent), self.arg() * exponent)
    }
}

impl Exp for ppga2d::Point {
    type Output = ppga2d::Motor;

    fn exp(self) -> ppga2d::Motor {
        let det = self.g0[0] * self.g0[0];
        if det <= 0.0 {
            return ppga2d::Motor {
                g0: simd::Simd32x4 {
                    f32x4: [1.0, 0.0, self.g0[1], self.g0[2]],
                },
            };
        }
        let a = det.sqrt();
        let c = a.cos();
        let s = a.sin() / a;
        let g0 = simd::Simd32x3::from(s) * self.g0;
        ppga2d::Motor {
            g0: simd::Simd32x4 {
                f32x4: [c, g0[0], g0[1], g0[2]],
            },
        }
    }
}

impl Ln for ppga2d::Motor {
    type Output = ppga2d::Point;

    fn ln(self) -> ppga2d::Point {
        let det = 1.0 - self.g0[0] * self.g0[0];
        if det <= 0.0 {
            return ppga2d::Point {
                g0: simd::Simd32x3 {
                    f32x3: [0.0, self.g0[2], self.g0[3]],
                },
            };
        }
        let a = 1.0 / det;
        let b = self.g0[0].acos() * a.sqrt();
        let g0 = simd::Simd32x4::from(b) * self.g0;
        return ppga2d::Point {
            g0: simd::Simd32x3 {
                f32x3: [g0[1], g0[2], g0[3]],
            },
        };
    }
}

impl Powf for ppga2d::Motor {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        (ppga2d::Scalar { g0: exponent } * self.ln()).exp()
    }
}

impl Exp for ppga3d::Line {
    type Output = ppga3d::Motor;

    fn exp(self) -> ppga3d::Motor {
        let det = self.g1[0] * self.g1[0] + self.g1[1] * self.g1[1] + self.g1[2] * self.g1[2];
        if det <= 0.0 {
            return ppga3d::Motor {
                g0: simd::Simd32x4 {
                    f32x4: [1.0, 0.0, 0.0, 0.0],
                },
                g1: simd::Simd32x4 {
                    f32x4: [0.0, self.g0[0], self.g0[1], self.g0[2]],
                },
            };
        }
        let a = det.sqrt();
        let c = a.cos();
        let s = a.sin() / a;
        let m = self.g0[0] * self.g1[0] + self.g0[1] * self.g1[1] + self.g0[2] * self.g1[2];
        let t = m / det * (c - s);
        let g0 = simd::Simd32x3::from(s) * self.g1;
        let g1 = simd::Simd32x3::from(s) * self.g0 + simd::Simd32x3::from(t) * self.g1;
        ppga3d::Motor {
            g0: simd::Simd32x4 {
                f32x4: [c, g0[0], g0[1], g0[2]],
            },
            g1: simd::Simd32x4 {
                f32x4: [s * m, g1[0], g1[1], g1[2]],
            },
        }
    }
}

impl Ln for ppga3d::Motor {
    type Output = ppga3d::Line;

    fn ln(self) -> ppga3d::Line {
        let det = 1.0 - self.g0[0] * self.g0[0];
        if det <= 0.0 {
            return ppga3d::Line {
                g0: simd::Simd32x3 {
                    f32x3: [self.g1[1], self.g1[2], self.g1[3]],
                },
                g1: simd::Simd32x3 {
                    f32x3: [0.0, 0.0, 0.0],
                },
            };
        }
        let a = 1.0 / det;
        let b = self.g0[0].acos() * a.sqrt();
        let c = a * self.g1[0] * (1.0 - self.g0[0] * b);
        let g0 = simd::Simd32x4::from(b) * self.g1 + simd::Simd32x4::from(c) * self.g0;
        let g1 = simd::Simd32x4::from(b) * self.g0;
        return ppga3d::Line {
            g0: simd::Simd32x3 {
                f32x3: [g0[1], g0[2], g0[3]],
            },
            g1: simd::Simd32x3 {
                f32x3: [g1[1], g1[2], g1[3]],
            },
        };
    }
}

impl Powf for ppga3d::Motor {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        (ppga3d::Scalar { g0: exponent } * self.ln()).exp()
    }
}

impl Exp for ppga3d::Branch {
    type Output = ppga3d::Translator;

    fn exp(self) -> ppga3d::Translator {
        ppga3d::Translator {
            g0: simd::Simd32x4 {
                f32x4: [1.0, self.g0[0], self.g0[1], self.g0[2]],
            }
        }
    }
}

impl Ln for ppga3d::Translator {
    type Output = ppga3d::Branch;

    fn ln(self) -> ppga3d::Branch {
        ppga3d::Branch {
            g0: simd::Simd32x3 {
                f32x3: [self.g0[1] / self.g0[0], self.g0[2] / self.g0[0], self.g0[3] / self.g0[0]],
            }
        }
    }
}

impl Powf for ppga3d::Translator {
    type Output = Self;

    fn powf(self, exponent: f32) -> Self {
        (ppga3d::Scalar { g0: exponent } * self.ln()).exp()
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
