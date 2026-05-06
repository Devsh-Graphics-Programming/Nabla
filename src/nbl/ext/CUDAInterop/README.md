# CUDA Interop Targets

- `Nabla::Nabla` stays CUDA-free. `find_package(Nabla CONFIG)` does not require the CUDA SDK.
- `Nabla::ext::CUDAInterop` is the clean Nabla interop target. Its public headers do not include `cuda.h` or `nvrtc.h`, so consumers can use a CUDA-enabled Nabla package without installing the CUDA SDK.
- `Nabla::ext::CUDAInteropNative` is the explicit raw CUDA opt-in target. It exposes `CUDAInteropNative.h`, CUDA Driver API and NVRTC types, and requires `CUDAToolkit`.
- Consumers can request native CUDA with `find_package(Nabla CONFIG COMPONENTS Core CUDAInteropNative)` and override the SDK root with `-DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>`.
- A consumer can use a newer compatible local CUDA SDK through `CUDAInteropNative` without rebuilding Nabla or the clean `CUDAInterop` target.
- Rebuilds stay local: changing CUDA SDK headers affects only targets that include `CUDAInteropNative.h`.

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
