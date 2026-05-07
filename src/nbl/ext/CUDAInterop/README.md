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

## OpenCV Reference

- OpenCV's common CUDA header includes OpenCV headers, not raw CUDA SDK headers: [`cuda.hpp`](https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/cuda.hpp#L51-L52).
- OpenCV keeps the public stream type as an OpenCV abstraction and grants access through `StreamAccessor`: [`cuda.hpp`](https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/cuda.hpp#L916-L979).
- OpenCV's raw CUDA opt-in header says it is the only header that depends on the CUDA Runtime API, then includes `<cuda_runtime.h>` and exposes accessor types: [`cuda_stream_accessor.hpp`](https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/cuda_stream_accessor.hpp#L50-L79).
- OpenCV also keeps implementation CUDA headers private and includes `<cuda.h>` / `<cuda_runtime.h>` there: [`private.cuda.hpp`](https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/private.cuda.hpp#L47-L61).
- The same split is used here: Nabla CUDA objects stay in `Nabla::Nabla`, and raw CUDA handles/functions are available only after including `CUDAInteropNative.h` and linking `Nabla::ext::CUDAInterop`.

## Usage

```cmake
find_package(Nabla CONFIG REQUIRED)
target_link_libraries(app PRIVATE Nabla::Nabla)
```

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS Core CUDAInterop)
target_link_libraries(native_app PRIVATE Nabla::ext::CUDAInterop)
```

## Properties

- `Nabla::Nabla` can be built with CUDA support without making CUDA SDK headers a public compile-time requirement.
- Consumers that only link `Nabla::Nabla` do not need a CUDA SDK to parse Nabla headers.
- Consumers that need raw CUDA include `CUDAInteropNative.h` and link `Nabla::ext::CUDAInterop` explicitly.
- Raw CUDA access is not wrapped away. Native code can use CUDA Driver API types, NVRTC types, and Nabla native accessors in the opt-in path.
- The Nabla source list is stable. CUDA interop `.cpp` files stay visible in IDE projects for CUDA ON and CUDA OFF builds.
- CUDA OFF implementations are local stubs in the same `.cpp` files. Clean API entry points stay linkable and return `nullptr` for unavailable CUDA features instead of producing unresolved symbols.
- CUDA implementation headers and SDK includes stay behind `_NBL_COMPILE_WITH_CUDA_`, so CUDA OFF builds do not need `cuda.h` or `nvrtc.h`.
- A package built with CUDA support can be consumed without a local CUDA SDK unless the `CUDAInterop` component is requested.
- A consumer can use a compatible local CUDA SDK for native interop without rebuilding Nabla.
