# CUDA Interop Targets

- `Nabla::Nabla` does not require the CUDA SDK.
- `Nabla::ext::CUDAInterop` provides Nabla CUDA interop types. Its public headers do not include `cuda.h` or `nvrtc.h`.
- `Nabla::ext::CUDAInteropNative` provides raw CUDA Driver API and NVRTC access through `CUDAInteropNative.h`.
- `CUDAInteropNative` requires `CUDAToolkit`. `CUDAInterop` does not expose that requirement to consumers.
- Consumers can override the SDK root with `-DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>` when requesting `CUDAInteropNative`.
- Consumers can build native CUDA code against a compatible local SDK without rebuilding Nabla or `CUDAInterop`.
- Changing CUDA SDK headers affects only targets that include `CUDAInteropNative.h`.
- Native accessors accept Nabla objects, raw pointers, and `smart_refctd_ptr`.

## Design

- CUDA is used privately while building the interop library.
- CUDA SDK headers become visible to consumers only through `CUDAInteropNative`.
- `CUDAInterop` exposes Nabla concepts such as devices, exported memory, imported memory, and imported semaphores.
- `CUDAInteropNative` exposes CUDA types such as `CUdeviceptr`, `CUmodule`, `CUfunction`, external memory, external semaphores, and NVRTC objects.
- The target split follows the same general dependency shape used by libraries such as OpenCV: common CUDA-facing APIs do not force raw CUDA headers on every consumer, while raw CUDA access is available through an explicit opt-in header.
- This avoids a transitive public compile-time dependency on CUDA from `Nabla::Nabla`.

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
