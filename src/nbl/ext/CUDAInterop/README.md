# CUDA Interop Targets

- `Nabla::Nabla` does not require the CUDA SDK.
- `Nabla::Nabla` provides Nabla CUDA interop types when the package was built with CUDA support.
- Nabla CUDA interop public headers do not include `cuda.h` or `nvrtc.h`.
- `Nabla::ext::CUDAInterop` is the raw CUDA Driver API and NVRTC opt-in target.
- `Nabla::ext::CUDAInterop` requires `CUDAToolkit` and exposes `CUDAInteropNative.h`.
- Consumers can override the SDK root with `-DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>` when requesting `CUDAInterop`.
- Consumers can build native CUDA code against a compatible local SDK without rebuilding Nabla.
- Changing CUDA SDK headers affects only targets that include `CUDAInteropNative.h`.
- Native accessors accept Nabla objects, raw pointers, and `smart_refctd_ptr`.

## Design

- CUDA is used privately while building `Nabla::Nabla`.
- CUDA SDK headers become visible to consumers only through `Nabla::ext::CUDAInterop`.
- `Nabla::Nabla` exposes Nabla concepts such as devices, exported memory, imported memory, and imported semaphores.
- `Nabla::ext::CUDAInterop` exposes CUDA types such as `CUdeviceptr`, `CUmodule`, `CUfunction`, external memory, external semaphores, and NVRTC objects.
- The dependency shape follows the same general model used by libraries such as OpenCV: common CUDA-facing APIs do not force raw CUDA headers on every consumer, while raw CUDA access is available through an explicit opt-in header.
- This avoids a transitive public compile-time dependency on CUDA from `Nabla::Nabla`.

## Usage

```cmake
find_package(Nabla CONFIG REQUIRED)
target_link_libraries(app PRIVATE Nabla::Nabla)
```

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS Core CUDAInterop)
target_link_libraries(native_app PRIVATE Nabla::ext::CUDAInterop)
```
