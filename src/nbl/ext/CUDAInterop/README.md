# CUDA Interop Targets

- `Nabla::Nabla` owns the CUDA interop implementation and exported symbols.
- `Nabla::Nabla` public headers do not include `cuda.h` or `nvrtc.h`.
- The SDK-free interop headers stay stable for CUDA ON and CUDA OFF Nabla builds.
- `Nabla::ext::CUDAInterop` is the explicit raw CUDA Driver API and NVRTC opt-in target.
- `Nabla::ext::CUDAInterop` is an `INTERFACE` target. It does not build a library or executable artifact.
- The target only carries usage requirements and IDE-visible sources.
- `Nabla::ext::CUDAInterop` requires `CUDAToolkit` and exposes `CUDAInteropNative.h`.
- `CUDAInteropNative.h` is the small opt-in header that includes CUDA SDK headers such as `cuda.h` and `nvrtc.h`.
- Consumers can override the SDK root with `-DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>` when requesting `CUDAInterop`.
- Native accessors accept Nabla objects, raw pointers, and `smart_refctd_ptr`.

## Basic Usage

```cmake
find_package(Nabla CONFIG REQUIRED)
target_link_libraries(app PRIVATE Nabla::Nabla)
```

This path does not require CUDA SDK headers on the consuming project.

## Native Opt-In

Use the native opt-in path only in targets that include `CUDAInteropNative.h` or use raw CUDA Driver API/NVRTC types.

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS CUDAInterop)
nbl_target_link_cuda_interop(native_app PRIVATE)
```

`nbl_target_link_cuda_interop` links `Nabla::ext::CUDAInterop` and writes runtime CUDA header discovery JSON for `native_app`.

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS CUDAInterop)
nbl_target_link_cuda_interop(native_app PRIVATE
    INCLUDE_DIRS "${cuda_runtime_headers}"
)
```

```cmake
nbl_target_link_cuda_interop(native_app PRIVATE
    RUNTIME_JSON "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/my_cuda_runtime.json"
)
```

Pseudo flow:

```cpp
#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"

auto handler = nbl::video::CCUDAHandler::create(system, std::move(logger));
auto cudaDevice = handler->createDevice(std::move(vulkanConnection), physicalDevice);

auto memory = nbl::video::cuda_native::createExportableMemory(cudaDevice, {
    .size = size,
    .alignment = alignment,
    .location = CU_MEM_LOCATION_TYPE_DEVICE,
});

std::string log;
auto [ptx, result] = nbl::video::cuda_native::compileDirectlyToPTX(
    handler,
    cudaSource,
    "kernel.cu",
    cudaDevice->geDefaultCompileOptions(),
    0,
    nullptr,
    nullptr,
    &log
);
```

`compileDirectlyToPTX` performs runtime CUDA header discovery internally. Code that drives NVRTC manually can call `cuda_interop::findRuntimeCompileEnvironment` and `cuda_interop::makeNVRTCIncludeOptions` directly.

Reference smoke:

- CMake target setup: `src/nbl/ext/CUDAInterop/smoke/CMakeLists.txt`
- SDK-free package boundary check: `src/nbl/ext/CUDAInterop/smoke/public_boundary.cpp`
- Default Nabla package usage without native opt-in: `src/nbl/ext/CUDAInterop/smoke/clean_opt_in.cpp`
- Native CUDA opt-in, runtime header discovery, `cuda_fp16.h`, NVRTC and raw interop usage: `src/nbl/ext/CUDAInterop/smoke/native_opt_in.cpp`

## Runtime Header Discovery

- `nbl_target_link_cuda_interop(<target> <scope>)` links `Nabla::ext::CUDAInterop` and configures runtime include discovery for that target.
- The helper is defined once in `NablaCUDAInteropHelpers.cmake` and is available from the source tree and installed `NablaConfig.cmake`.
- For each target it writes `nbl_cuda_interop_runtime.json` next to the executable during CMake generation.
- `RUNTIME_JSON <path>` overrides the generated JSON location. Plain paths and `$<CONFIG>` are supported.
- `cuda_interop::findRuntimeCompileEnvironment` can also receive explicit JSON paths at runtime.
- `NBL_CUDA_INTEROP_RUNTIME_JSON` can point runtime discovery at custom JSON files without rebuilding the application.
- The JSON is a build artifact. Nabla packages do not install JSON files with host-specific CUDA paths.
- Package consumers generate their own JSON when they call `nbl_target_link_cuda_interop`.
- Runtime lookup reads explicit JSON paths and `NBL_CUDA_INTEROP_RUNTIME_JSON` first, then checks executable-local `nbl_cuda_interop_runtime.json`, app-local include bundles, explicit include-dir environment variables, `CUDA_PATH` style toolkit roots, Python/conda package layouts, and common system install roots.
- App-local and Python/conda package probing looks for directories that contain CUDA runtime headers. It does not hardcode a CUDA major version in the path.
- `cuda_native::compileDirectlyToPTX` appends discovered include directories to the NVRTC option list and caches the default discovery result after first use.
- Production machines do not need the full CUDA SDK just because Nabla was built with CUDA.
- If an application compiles CUDA source with NVRTC and includes headers such as `cuda_fp16.h`, it must provide those runtime headers through the generated JSON path, an app-local bundle, a runtime/header package, or an installed toolkit.
- `CUDA_PATH` is a developer fallback. It is not required for packaged applications.
- Direct `target_link_libraries(app PRIVATE Nabla::ext::CUDAInterop)` remains possible, but it only adds compile/link usage requirements and does not create the runtime discovery JSON.

## Runtime Header Distribution

Nabla packages do not ship CUDA runtime headers. That is a packaging choice, not a hard legal requirement for applications that need NVRTC runtime compilation.

NVIDIA CUDA EULA limits CUDA redistribution to selected components. The distribution section says: "The portions of the SDK that are distributable under the Agreement are listed in Attachment A." Attachment A then lists the CUDA Toolkit files that may be redistributed with applications. See:

- https://docs.nvidia.com/cuda/eula/#distribution
- https://docs.nvidia.com/cuda/eula/#attachment-a

Relevant Attachment A header entries include:

- `nvrtc.h` under `NVIDIA Runtime Compilation Library and Header`.
- `cuda_occupancy.h` under `CUDA Occupancy Calculation Header Library`.
- `cuda_fp16.h`, `cuda_fp16.hpp`, `cuda_bf16.h`, `cuda_bf16.hpp`, `cuda_fp8.h`, `cuda_fp8.hpp`, `cuda_fp6.h`, `cuda_fp6.hpp`, `cuda_fp4.h`, `cuda_fp4.hpp` under `CUDA Floating Point Type Headers`.
- `crt/host_defines.h`, `cuComplex.h`, `cuda_awbarrier_helpers.h`, `cuda_awbarrier_primitives.h`, `cuda_awbarrier.h`, `cuda_pipeline_helpers.h`, `cuda_pipeline_primitives.h`, `cuda_pipeline.h`, `cuda_runtime_api.h`, `cuda.h`, `cuda/std/tuple`, `cuda/std/type_traits`, `cuda/std/utility`, `device_types.h`, `vector_functions.h`, `vector_types.h` under `CUDA Headers for Runtime Compilation`.

CuPy documents the same runtime-compile problem. Their install docs say: "On CUDA 12.2 or later, CUDA Runtime header files are required to compile kernels in CuPy." They also show the common `vector_types.h` failure and recommend `nvidia-cuda-runtime-cu12` for PyPI installs or `cuda-cudart-dev` from system packages:

- https://docs.cupy.dev/en/v13.5.0/install.html#cupy-always-raises-nvrtc-error-compilation-6
- https://github.com/cupy/cupy/issues/8466

For Nabla consumers this means:

- The default Nabla package stays SDK-free for consumers that only link `Nabla::Nabla`.
- Native interop consumers can install CUDA runtime headers through an official package, point `NBL_CUDA_INTEROP_RUNTIME_JSON` at their own JSON, pass `INCLUDE_DIRS` to `nbl_target_link_cuda_interop`, or ship an app-local header bundle if their distribution model allows it.
- Shipping such headers is a consumer packaging decision. Nabla runtime discovery supports it, but Nabla does not install host-specific CUDA header paths or redistribute CUDA headers by default.

## Properties

- Consumers that only link `Nabla::Nabla` do not need CUDA SDK headers to parse Nabla headers.
- Consumers that need raw CUDA include `CUDAInteropNative.h` and link `Nabla::ext::CUDAInterop`.
- Raw CUDA access is not wrapped away in the native opt-in path. Native code uses CUDA Driver API and NVRTC types directly.
- CUDA SDK structs with version-sensitive layout are kept out of exported Nabla ABI.
- The exported native ABI uses stable CUDA Driver API handles/enums and small Nabla-owned parameter structs.
- Native state is PIMPL-owned by Nabla. Consumers cannot construct CUDA wrapper objects with arbitrary internal state.
- A package built with one compatible CUDA SDK can be consumed by native interop code built with another compatible SDK without rebuilding Nabla.
- `CCUDAHandler::create` validates the loaded CUDA driver and NVRTC runtime. It returns `nullptr` when the runtime is missing or below the required CUDA 13.0 / NVRTC 13.x floor.
- Runtime CUDA header discovery is independent from the CUDA SDK used to build Nabla.
- Native consumers can use a newer compatible CUDA SDK or a runtime/header package without rebuilding Nabla.
- Toggling Nabla CUDA support does not change SDK-free public header parse requirements for consumers.
- The Nabla source list is stable. CUDA interop `.cpp` files stay visible in IDE projects for CUDA ON and CUDA OFF builds.
- CUDA OFF implementations are local stubs in the same `.cpp` files. SDK-free API entry points stay linkable and factory/import/export paths return `nullptr` for unavailable CUDA features instead of producing unresolved symbols.
- CUDA implementation headers and SDK includes stay behind `_NBL_COMPILE_WITH_CUDA_`.

## Related Designs

This split follows the same public-boundary pattern used by mature GPU projects: SDK-free default headers, native access through an explicit opt-in path, and SDK-dependent implementation details outside the default public API.

- OpenCV keeps common CUDA-facing headers independent from CUDA Runtime API and exposes raw `cudaStream_t` / `cudaEvent_t` through a separate accessor header: [`cuda_stream_accessor.hpp`](https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/cuda_stream_accessor.hpp#L50-L79).
- OpenCV keeps CUDA implementation headers private and includes `cuda.h`, `cuda_runtime.h`, and NPP there: [`private.cuda.hpp`](https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/private.cuda.hpp#L47-L61).
- Blender/Cycles exposes a CUDA device boundary without CUDA SDK headers in the boundary header: [`device.h`](https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device.h#L7-L27).
- Blender/Cycles keeps `CUdevice`, `CUcontext`, `cuda.h`, and `cuew.h` in the CUDA implementation header/source: [`device_impl.h`](https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device_impl.h#L12-L30), [`device.cpp`](https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device.cpp#L10-L48).
- ONNX Runtime keeps accelerator dependencies behind execution providers and supports provider shared libraries loaded only when requested: [`Build with Execution Providers`](https://onnxruntime.ai/docs/build/eps.html#execution-provider-shared-libraries).
- ggml/llama.cpp keeps the generic backend API separate from CUDA and builds CUDA as an explicit backend target with CUDA libraries linked to that backend: [`ggml-backend.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h#L1488-L1499), [`ggml-cuda CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt#L982-L1072).
- TensorFlow PluggableDevice uses separate device plugin packages so accelerator toolchains and dependencies do not become core TensorFlow requirements: [`PluggableDevice`](https://blog.tensorflow.org/2021/06/pluggabledevice-device-plugins-for-TensorFlow.html).
