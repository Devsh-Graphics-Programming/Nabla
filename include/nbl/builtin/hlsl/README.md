# Nabla HLSL Builtin Library

## Overview

Shared shader code for Nabla. Many headers compile both as HLSL on the GPU and
as C++ on the host. The two paths are split by the `__HLSL_VERSION` macro: the
host branch often forwards to the C++ standard library, while the device branch
provides the GPU equivalent, often backed by SPIR-V intrinsics. `bit.hlsl` is a
small example of both branches in one file.

All code lives in the `nbl::hlsl` namespace.

## Single-source HLSL/C++

This folder is Nabla's shader-side support library: reusable HLSL headers for
math, types, intrinsics, algorithms, compatibility helpers, and graphics code.
The same header can be compiled by the shader compiler for GPU use and by the
C++ compiler for host-side tests, tools, and shared data definitions.

The compatibility layer is what makes that possible. `cpp_compat.hlsl` and the
`cpp_compat/` folder provide the C++ side of HLSL types and intrinsics, while
other headers use `__HLSL_VERSION` to choose between host implementations
(`std`, C++ helpers, or test code) and device implementations (HLSL builtins,
GLSL-style compatibility wrappers, or SPIR-V intrinsics).

This lets Nabla write shader abstractions in HLSL while keeping them close
enough to C++ to share structures, validate behavior on the host, and build
shader utilities similar to parts of the C++ standard library.

For more context, see the DevSH presentations:

- https://www.youtube.com/watch?v=JCJ35dlZJb4
- https://www.devsh.eu/presentations

## Placement rules

Root files that share names with C++ standard library headers are for the
matching `std::` counterpart only. For example, `bit.hlsl` is the `<bit>`
counterpart and should stay limited to things like `bit_cast`, `rotl`, `rotr`,
and `countl_zero`.

Code beyond the standard library goes in a subfolder, even if it is related to a
root file. The bitfield abstraction is not part of `<bit>`, so it belongs in
`utils/bitfield.hlsl`, not in `bit.hlsl`.

Prefer folder names that match their namespaces, but do not force it when the
result reads badly. `utils/bitfield.hlsl` is better than
`bitfield/bitfield.hlsl` if the second form pushes toward
`hlsl::bitfield::bitfield`.

## How to extend it

1. If it mirrors a C++ standard library header, put it in the matching root
   file.
2. If it is a Nabla extension or helper, put it in a subfolder.
3. Reuse an existing folder when it fits. Add a new one only when needed.

## Structure

### Root files

The "std" column marks files that are the GPU counterpart of a C++ standard
library header. Those hold only what the standard header provides.

| File | std | Contents and what belongs here |
| --- | --- | --- |
| `macros.h` | | Core preprocessor macros, including the `static_assert`/`assert` shims for HLSL. Cross-compilation macros go here. |
| `cpp_compat.hlsl` | | Umbrella include for the C++ compatibility layer (pulls in `cpp_compat/`). |
| `algorithm.hlsl` | `<algorithm>` | `swap` and other standard algorithms. |
| `bit.hlsl` | `<bit>` | `bit_cast`, `rotl`, `rotr`, `countl_zero`. |
| `complex.hlsl` | `<complex>` | `complex_t`. |
| `concepts.hlsl` | `<concepts>` | Entry point for the concepts library (includes `concepts/`). |
| `functional.hlsl` | `<functional>` | `reference_wrapper` and function objects. |
| `limits.hlsl` | `<limits>` | `numeric_limits`. |
| `memory.hlsl` | `<memory>` | Pointer and reference helpers, e.g. `pointer_to` for BDA refs. |
| `numbers.hlsl` | `<numbers>` | Math constants (`e`, `pi`, ...). |
| `tuple.hlsl` | `<tuple>` | `tuple`. |
| `type_traits.hlsl` | `<type_traits>` | Type trait structs and aliases. |
| `utility.hlsl` | `<utility>` | Standard utilities (currently `declval`). |
| `tgmath.hlsl` | `<tgmath>`/`<cmath>` | Type-generic math functions. |
| `ieee754.hlsl` | | IEEE-754 float layout traits and bit helpers. |
| `mpl.hlsl` | | Template metaprogramming helpers (compile-time, boost::mpl style). |
| `enums.hlsl` | | Engine enums shared host and device (e.g. `ShaderStage`). |
| `format.hlsl` | | Texel block format enum and format pack/unpack (includes `format/`). |
| `colorspace.hlsl` | | Colorspace conversions entry point (includes `colorspace/`). |
| `morton.hlsl` | | Morton / Z-order code encode and decode. |
| `acceleration_structures.hlsl` | | Ray tracing acceleration structure build structs. |
| `indirect_commands.hlsl` | | Indirect draw and dispatch command structs. |
| `binding_info.hlsl` | | Compile-time descriptor binding info structs. |
| `device_capabilities_traits.hlsl` | | Traits to query device capabilities at compile time. |
| `array_accessors.hlsl` | | Generic array `get`/`set` accessor structs. |
| `memory_accessor.hlsl` | | Accessor wrappers over memory with atomic and barrier method detection. |
| `member_test_macros.hlsl` | | Macros to detect presence of struct members and methods. |
| `ndarray_addressing.hlsl` | | Multi-dimensional to linear index addressing. |
| `scanning_append.hlsl` | | Result types for the scan-and-append primitive. |
| `surface_transform.h` | | Swapchain surface transform flags and helpers. |

### Subfolders

| Folder | Contents and what belongs here |
| --- | --- |
| `barycentric/` | Barycentric coordinate utilities. |
| `bda/` | Buffer Device Address: typed pointers, references, and accessors. |
| `blit/` | Image blit and normalization compute shaders and their parameters. |
| `bxdf/` | BxDF models: fresnel, NDF, reflection, transmission. |
| `colorspace/` | Transfer functions (EOTF/OETF) and CIEXYZ encode/decode. |
| `concepts/` | Concept definitions (core, vector, matrix, accessors). |
| `cpp_compat/` | The C++/HLSL compatibility layer: vector, matrix, intrinsics, promote, truncate. |
| `emulated/` | Software-emulated types (`float64_t`, `int64_t`, emulated vector/matrix) for platforms without native support. |
| `ext/` | Helpers tied to engine extensions (e.g. FullScreenTriangle). |
| `fft/` | FFT building blocks. See [`fft/README.md`](fft/README.md). |
| `format/` | Pixel/texel format pack and unpack (octahedral, shared exponent). |
| `glsl_compat/` | GLSL builtin equivalents (core, subgroup ops). |
| `ieee754/` | IEEE-754 implementation details. |
| `ies/` | IES light profile sampling and textures. |
| `math/` | Math routines: geometry, linalg, quaternions, equations, quadrature, and more. |
| `matrix_utils/` | Matrix traits, compile-time and runtime. |
| `path_tracing/` | Path tracing building blocks: ray gen, accumulators, integrators. |
| `portable/` | Type aliases that pick native or emulated types per platform. |
| `prefix_sum_blur/` | Prefix-sum based blur. |
| `random/` | RNGs (lcg, pcg, tea, xoroshiro) and adaptors. |
| `rwmc/` | Reweighted Monte Carlo: cascade accumulator, resolve, splatting. |
| `sampling/` | Sampling distributions and warps (alias table, spherical shapes, mappings). |
| `scan/` | Device-wide scan (prefix sum) primitives and schedulers. |
| `shapes/` | Geometric primitives (aabb, obb, line, triangle, beziers, ...). |
| `sort/` | Sorting primitives (counting sort). |
| `spirv_intrinsics/` | Raw SPIR-V intrinsic declarations. |
| `subgroup/` | Subgroup-level collectives (ballot, arithmetic, basic, fft). |
| `subgroup2/` | Newer subgroup collective API. |
| `testing/` | Comparison helpers for tests (approx compare, max error). |
| `text_rendering/` | Text rendering (MSDF). |
| `tgmath/` | `tgmath` implementation details (isnan, output structs). |
| `vector_utils/` | Vector traits. |
| `visualization/` | Visualization helpers (turbo colormap). |
| `workgroup/` | Workgroup-level collectives (arithmetic, ballot, scan, shuffle, fft). |
| `workgroup2/` | Newer workgroup collective API. |
