# CUDA Interop Targets

This extension keeps CUDA interop available without making CUDA a default public
compile-time dependency of Nabla.

- `Nabla::Nabla` stays CUDA-free. `find_package(Nabla CONFIG)` does not require the CUDA SDK.
- `Nabla::ext::CUDAInterop` is the clean Nabla interop target. Its public headers do not include `cuda.h` or `nvrtc.h`, so consumers can use a CUDA-enabled Nabla package without installing the CUDA SDK.
- `Nabla::ext::CUDAInteropNative` is the explicit raw CUDA opt-in target. It exposes `CUDAInteropNative.h`, CUDA Driver API and NVRTC types, and requires `CUDAToolkit`.
- Consumers can request native CUDA with `find_package(Nabla CONFIG COMPONENTS Core CUDAInteropNative)` and override the SDK root with `-DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>`.
- A consumer can use a newer compatible local CUDA SDK through `CUDAInteropNative` without rebuilding Nabla or the clean `CUDAInterop` target.
- Rebuilds stay local: changing CUDA SDK headers affects only targets that include `CUDAInteropNative.h`.
- Native accessors accept Nabla objects, raw pointers, and `smart_refctd_ptr`, so opt-in code can keep CUDA usage terse without moving CUDA types into clean headers.

## Design

- The default Nabla package remains relocatable and usable on machines without the CUDA SDK.
- CUDA is used privately to build the interop library. CUDA SDK headers become visible to consumers only when `CUDAInteropNative` is requested.
- Clean interop headers expose Nabla concepts such as devices, exported memory, imported memory, and imported semaphores.
- Native interop headers expose raw CUDA Driver API and NVRTC types for examples and applications that need direct CUDA work.
- The split is intentionally similar to the OpenCV CUDA shape: common CUDA-facing headers stay clean, while raw CUDA access lives behind explicit opt-in accessor/native headers.
- This avoids a transitive public compile-time dependency on CUDA while preserving the low-level workflow for kernels, `CUdeviceptr`, `CUmodule`, `CUfunction`, external memory, and external semaphores.
- Package consumers can pick their own compatible CUDA SDK for native code without rebuilding Nabla or the clean interop library.

## Usage

```cmake
find_package(Nabla CONFIG REQUIRED)
target_link_libraries(app PRIVATE Nabla::Nabla)
```

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS Core CUDAInterop)
target_link_libraries(app PRIVATE Nabla::ext::CUDAInterop)
```

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS Core CUDAInteropNative)
target_link_libraries(app PRIVATE Nabla::ext::CUDAInteropNative)
```
