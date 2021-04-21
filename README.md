[![actions](https://github.com/Lichtso/geometric_algebra/actions/workflows/actions.yaml/badge.svg)](https://github.com/Lichtso/geometric_algebra/actions/workflows/actions.yaml)
[![Docs](https://docs.rs/geometric_algebra/badge.svg)](https://docs.rs/geometric_algebra/)
[![crates.io](https://img.shields.io/crates/v/geometric_algebra.svg)](https://crates.io/crates/geometric_algebra)

## About
This repository allows you to describe [geometric algebras](https://en.wikipedia.org/wiki/Geometric_algebra) with 1 to 16 generator elements and generate SIMD-ready, dependency-less libraries for them. It also comes with a set of prebuilt projective geometric algebras in 1D, 2D and 3D which are elliptic, parabolic (euclidian) or hyperbolic.

## Architecture
- [DSL](https://en.wikipedia.org/wiki/Domain-specific_language) Parser: See [examples](.github/workflows/actions.yaml)
- Algebra: Generates the multiplication tables
- Compiler: Constructs an AST from the multiplication tables
- Optimizer: Simplifies the AST
- Legalizer: Inserts missing expressions in the AST
- Emitter: Serializes the AST to source code
    - [Rust](https://www.rust-lang.org/)
    - [GLSL](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))
