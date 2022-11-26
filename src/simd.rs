#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use std::arch::aarch64::*;
#[cfg(all(target_arch = "arm", target_feature = "neon"))]
pub use std::arch::arm::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use std::arch::wasm32::*;
#[cfg(all(target_arch = "x86", target_feature = "sse2"))]
pub use std::arch::x86::*;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub use std::arch::x86_64::*;

#[derive(Clone, Copy)]
#[repr(C)]
pub union Simd32x4 {
    // Intel / AMD
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    pub f128: __m128,
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    pub i128: __m128i,

    // ARM
    #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
    pub f128: float32x4_t,
    #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
    pub i128: int32x4_t,
    #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
    pub u128: uint32x4_t,

    // Web
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub v128: v128,

    // Fallback
    pub f32x4: [f32; 4],
    pub i32x4: [i32; 4],
    pub u32x4: [u32; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union Simd32x3 {
    pub v32x4: Simd32x4,

    // Fallback
    pub f32x3: [f32; 3],
    pub i32x3: [i32; 3],
    pub u32x3: [u32; 3],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub union Simd32x2 {
    pub v32x4: Simd32x4,

    // Fallback
    pub f32x2: [f32; 2],
    pub i32x2: [i32; 2],
    pub u32x2: [u32; 2],
}

#[macro_export]
macro_rules! match_architecture {
    ($Simd:ident, $native:tt, $fallback:tt,) => {{
        #[cfg(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"),
            all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128"),
        ))]
        { $Simd $native }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"),
            all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128"),
        )))]
        unsafe { $Simd $fallback }
    }};
    ($Simd:ident, $x86:tt, $arm:tt, $web:tt, $fallback:tt,) => {{
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
        unsafe { $Simd $x86 }
        #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
        unsafe { $Simd $arm }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe { $Simd $web }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"),
            all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128"),
        )))]
        unsafe { $Simd $fallback }
    }};
}

#[macro_export]
macro_rules! swizzle {
    ($self:expr, $x:literal, $y:literal, $z:literal, $w:literal) => {
        $crate::match_architecture!(
            Simd32x4,
            { f128: $crate::simd::_mm_permute_ps($self.f128, ($x as i32) | (($y as i32) << 2) | (($z as i32) << 4) | (($w as i32) << 6)) },
            { f32x4: [
                $self.f32x4[$x],
                $self.f32x4[$y],
                $self.f32x4[$z],
                $self.f32x4[$w],
            ] },
            { v128: $crate::simd::i32x4_shuffle::<$x, $y, $z, $w>($self.v128, $self.v128) },
            { f32x4: [
                $self.f32x4[$x],
                $self.f32x4[$y],
                $self.f32x4[$z],
                $self.f32x4[$w],
            ] },
        )
    };
    ($self:expr, $x:literal, $y:literal, $z:literal) => {
        $crate::match_architecture!(
            Simd32x3,
            { v32x4: $crate::swizzle!($self.v32x4, $x, $y, $z, 0) },
            { f32x3: [
                $self.f32x3[$x],
                $self.f32x3[$y],
                $self.f32x3[$z],
            ] },
        )
    };
    ($self:expr, $x:literal, $y:literal) => {
        $crate::match_architecture!(
            Simd32x2,
            { v32x4: $crate::swizzle!($self.v32x4, $x, $y, 0, 0) },
            { f32x2: [
                $self.f32x2[$x],
                $self.f32x2[$y],
            ] },
        )
    };
}

impl std::ops::Index<usize> for Simd32x4 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.f32x4[index] }
    }
}

impl std::ops::Index<usize> for Simd32x3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.f32x3[index] }
    }
}

impl std::ops::Index<usize> for Simd32x2 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.f32x2[index] }
    }
}

impl std::ops::IndexMut<usize> for Simd32x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut self.f32x4[index] }
    }
}

impl std::ops::IndexMut<usize> for Simd32x3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut self.f32x3[index] }
    }
}

impl std::ops::IndexMut<usize> for Simd32x2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut self.f32x2[index] }
    }
}

impl std::convert::From<Simd32x4> for [f32; 4] {
    fn from(simd: Simd32x4) -> Self {
        unsafe { simd.f32x4 }
    }
}

impl std::convert::From<Simd32x3> for [f32; 3] {
    fn from(simd: Simd32x3) -> Self {
        unsafe { simd.f32x3 }
    }
}

impl std::convert::From<Simd32x2> for [f32; 2] {
    fn from(simd: Simd32x2) -> Self {
        unsafe { simd.f32x2 }
    }
}

impl std::convert::From<[f32; 4]> for Simd32x4 {
    fn from(f32x4: [f32; 4]) -> Self {
        Self { f32x4 }
    }
}

impl std::convert::From<[f32; 3]> for Simd32x3 {
    fn from(f32x3: [f32; 3]) -> Self {
        Self { f32x3 }
    }
}

impl std::convert::From<[f32; 2]> for Simd32x2 {
    fn from(f32x2: [f32; 2]) -> Self {
        Self { f32x2 }
    }
}

impl std::convert::From<f32> for Simd32x4 {
    fn from(value: f32) -> Self {
        Self {
            f32x4: [value, value, value, value],
        }
    }
}

impl std::convert::From<f32> for Simd32x3 {
    fn from(value: f32) -> Self {
        Self {
            f32x3: [value, value, value],
        }
    }
}

impl std::convert::From<f32> for Simd32x2 {
    fn from(value: f32) -> Self {
        Self {
            f32x2: [value, value],
        }
    }
}

impl std::fmt::Debug for Simd32x4 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .debug_list()
            .entries([self[0], self[1], self[2], self[3]].iter())
            .finish()
    }
}

impl std::fmt::Debug for Simd32x3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .debug_list()
            .entries([self[0], self[1], self[2]].iter())
            .finish()
    }
}

impl std::fmt::Debug for Simd32x2 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .debug_list()
            .entries([self[0], self[1]].iter())
            .finish()
    }
}

impl std::ops::Add<Simd32x4> for Simd32x4 {
    type Output = Simd32x4;

    fn add(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { f128: _mm_add_ps(self.f128, other.f128) },
            { f128: vaddq_f32(self.f128, other.f128) },
            { v128: f32x4_add(self.v128, other.v128) },
            { f32x4: [
                self.f32x4[0] + other.f32x4[0],
                self.f32x4[1] + other.f32x4[1],
                self.f32x4[2] + other.f32x4[2],
                self.f32x4[3] + other.f32x4[3],
            ] },
        )
    }
}

impl std::ops::Add<Simd32x3> for Simd32x3 {
    type Output = Simd32x3;

    fn add(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 + other.v32x4 } },
            { f32x3: [
                self.f32x3[0] + other.f32x3[0],
                self.f32x3[1] + other.f32x3[1],
                self.f32x3[2] + other.f32x3[2],
            ] },
        )
    }
}

impl std::ops::Add<Simd32x2> for Simd32x2 {
    type Output = Simd32x2;

    fn add(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 + other.v32x4 } },
            { f32x2: [
                self.f32x2[0] + other.f32x2[0],
                self.f32x2[1] + other.f32x2[1],
            ] },
        )
    }
}

impl std::ops::Sub<Simd32x4> for Simd32x4 {
    type Output = Simd32x4;

    fn sub(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { f128: _mm_sub_ps(self.f128, other.f128) },
            { f128: vsubq_f32(self.f128, other.f128) },
            { v128: f32x4_sub(self.v128, other.v128) },
            { f32x4: [
                self.f32x4[0] - other.f32x4[0],
                self.f32x4[1] - other.f32x4[1],
                self.f32x4[2] - other.f32x4[2],
                self.f32x4[3] - other.f32x4[3],
            ] },
        )
    }
}

impl std::ops::Sub<Simd32x3> for Simd32x3 {
    type Output = Simd32x3;

    fn sub(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 - other.v32x4 } },
            { f32x3: [
                self.f32x3[0] - other.f32x3[0],
                self.f32x3[1] - other.f32x3[1],
                self.f32x3[2] - other.f32x3[2],
            ] },
        )
    }
}

impl std::ops::Sub<Simd32x2> for Simd32x2 {
    type Output = Simd32x2;

    fn sub(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 - other.v32x4 } },
            { f32x2: [
                self.f32x2[0] - other.f32x2[0],
                self.f32x2[1] - other.f32x2[1],
            ] },
        )
    }
}

impl std::ops::Mul<Simd32x4> for Simd32x4 {
    type Output = Simd32x4;

    fn mul(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { f128: _mm_mul_ps(self.f128, other.f128) },
            { f128: vmulq_f32(self.f128, other.f128) },
            { v128: f32x4_mul(self.v128, other.v128) },
            { f32x4: [
                self.f32x4[0] * other.f32x4[0],
                self.f32x4[1] * other.f32x4[1],
                self.f32x4[2] * other.f32x4[2],
                self.f32x4[3] * other.f32x4[3],
            ] },
        )
    }
}

impl std::ops::Mul<Simd32x3> for Simd32x3 {
    type Output = Simd32x3;

    fn mul(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 * other.v32x4 } },
            { f32x3: [
                self.f32x3[0] * other.f32x3[0],
                self.f32x3[1] * other.f32x3[1],
                self.f32x3[2] * other.f32x3[2],
            ] },
        )
    }
}

impl std::ops::Mul<Simd32x2> for Simd32x2 {
    type Output = Simd32x2;

    fn mul(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 * other.v32x4 } },
            { f32x2: [
                self.f32x2[0] * other.f32x2[0],
                self.f32x2[1] * other.f32x2[1],
            ] },
        )
    }
}

impl std::ops::Div<Simd32x4> for Simd32x4 {
    type Output = Simd32x4;

    fn div(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { f128: _mm_div_ps(self.f128, other.f128) },
            { f128: vdivq_f32(self.f128, other.f128) },
            { v128: f32x4_div(self.v128, other.v128) },
            { f32x4: [
                self.f32x4[0] / other.f32x4[0],
                self.f32x4[1] / other.f32x4[1],
                self.f32x4[2] / other.f32x4[2],
                self.f32x4[3] / other.f32x4[3],
            ] },
        )
    }
}

impl std::ops::Div<Simd32x3> for Simd32x3 {
    type Output = Simd32x3;

    fn div(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 / other.v32x4 } },
            { f32x3: [
                self.f32x3[0] / other.f32x3[0],
                self.f32x3[1] / other.f32x3[1],
                self.f32x3[2] / other.f32x3[2],
            ] },
        )
    }
}

impl std::ops::Div<Simd32x2> for Simd32x2 {
    type Output = Simd32x2;

    fn div(self, other: Self) -> Self {
        match_architecture!(
            Self,
            { v32x4: unsafe { self.v32x4 / other.v32x4 } },
            { f32x2: [
                self.f32x2[0] / other.f32x2[0],
                self.f32x2[1] / other.f32x2[1],
            ] },
        )
    }
}
