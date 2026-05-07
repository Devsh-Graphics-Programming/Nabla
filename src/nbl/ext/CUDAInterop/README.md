# CUDA Interop

## Layout

- `Nabla::Nabla` owns the SDK-free CUDA interop API in `nbl/video/CCUDA*.h` and its implementation in `src/nbl/video/CCUDA*.cpp`.
- Those headers do not include CUDA SDK headers. Consumers that only link `Nabla::Nabla` do not need `cuda.h`, `nvrtc.h`, or a CUDA SDK install just to parse Nabla headers.
- `Nabla::ext::CUDAInterop` is an `INTERFACE` target for native CUDA opt-in. It builds no library. It only adds `CUDAInteropNative.h`, `CUDA::toolkit`, and runtime-header discovery setup to targets that ask for raw CUDA interop.
- `CUDAInteropNative.h` is the only public opt-in header that includes CUDA SDK headers and exposes `cuda_native::*Accessor` classes for CUDA Driver API and NVRTC types.

## CMake Usage

Default Nabla usage stays SDK-free:

```cmake
find_package(Nabla CONFIG REQUIRED)
target_link_libraries(app PRIVATE Nabla::Nabla)
```

Native CUDA interop is explicit:

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS CUDAInterop)
nbl_target_link_cuda_interop(native_app PRIVATE)
```

`nbl_target_link_cuda_interop` links `Nabla::ext::CUDAInterop` and writes `nbl_cuda_interop_runtime.json` next to the target executable during CMake generation.

Optional overrides:

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS CUDAInterop)
nbl_target_link_cuda_interop(native_app PRIVATE
    INCLUDE_DIRS "${cuda_runtime_headers}"
)

nbl_target_link_cuda_interop(native_app PRIVATE
    RUNTIME_JSON "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/my_cuda_runtime.json"
)
```

Consumers can also choose the SDK used for native compilation with:

```cmake
cmake -S . -B build -DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>
```

This affects native opt-in compilation and generated runtime header discovery only. It does not rebuild Nabla and does not change the `Nabla.dll` ABI.

## Native Usage

```cpp
#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"

auto handler = nbl::video::CCUDAHandler::create(system, std::move(logger));
auto cudaDevice = handler->createDevice(std::move(vulkanConnection), physicalDevice);

auto memory = nbl::video::cuda_native::CCUDADeviceAccessor::createExportableMemory(*cudaDevice, {
    .size = size,
    .alignment = alignment,
    .location = CU_MEM_LOCATION_TYPE_DEVICE,
});

std::string log;
std::string cudaSource = loadKernelText();
auto compile = nbl::video::cuda_native::CCUDAHandlerAccessor::compileDirectlyToPTX(
    *handler,
    std::move(cudaSource),
    "kernel.cu",
    cudaDevice->geDefaultCompileOptions(),
    log,
    0,
    nullptr,
    nullptr
);
```

Native access is not wrapped away. Opt-in code uses CUDA Driver API and NVRTC types directly through accessor classes:

- `CCUDAHandlerAccessor` exposes CUDA/NVRTC function tables, NVRTC program helpers, PTX compilation, native device enumeration, and default error handling.
- `CCUDADeviceAccessor` exposes `CUdevice`, `CUcontext`, memory granularity, and CUDA allocation creation.
- `CCUDAExportableMemoryAccessor`, `CCUDAImportedMemoryAccessor`, and `CCUDAImportedSemaphoreAccessor` expose the raw CUDA handles needed for interop.
- Accessor methods take explicit Nabla references. Callers dereference `smart_refctd_ptr` at the call site instead of going through pointer/smart-pointer convenience overloads.
- `compileDirectlyToPTX` returns PTX/result and writes the NVRTC log to a required `std::string&`. There is no optional output pointer in the public API.

Smoke examples:

- `src/nbl/ext/CUDAInterop/smoke/public_boundary.cpp` checks that `Nabla::Nabla` headers stay SDK-free.
- `src/nbl/ext/CUDAInterop/smoke/clean_opt_in.cpp` checks default package usage without native opt-in.
- `src/nbl/ext/CUDAInterop/smoke/native_opt_in.cpp` checks native opt-in, runtime header discovery, `cuda_fp16.h`, NVRTC, and raw interop usage.

## ABI

- `CCUDAHandler`, `CCUDADevice`, `CCUDAExportableMemory`, `CCUDAImportedMemory`, and `CCUDAImportedSemaphore` are exported from `Nabla.dll` through the normal Nabla ABI.
- Their public declarations do not expose CUDA SDK structs, CUDA SDK layouts, or `cuda.h` / `nvrtc.h` includes.
- CUDA implementation state is owned by Nabla through private `SNativeState` members. Consumers cannot construct CUDA wrapper objects with arbitrary internal CUDA state.
- `CUDAInteropNative.h` declares exported accessor classes whose definitions still live in `Nabla.dll`. The opt-in header owns only the CUDA SDK surface. Nabla owns the implementation and ABI.
- Native opt-in ABI uses CUDA Driver API handles/enums such as `CUdevice`, `CUcontext`, `CUdeviceptr`, `CUexternalMemory`, and `CUexternalSemaphore`, plus small fixed-layout parameter/result structs.
- SDK-sized arrays and other layouts derived from CUDA SDK constants stay private to Nabla. A consumer can build native opt-in code with its own compatible SDK independently from the SDK used to build Nabla.
- Runtime include-option construction is header-only and is not part of the exported ABI.
- The loaded CUDA driver and NVRTC runtime are validated at runtime.

## Runtime Header Discovery

NVRTC may need CUDA runtime headers when user kernels include files such as `cuda_fp16.h`, `vector_types.h`, or `cuda_runtime_api.h`. This is a runtime concern of applications that compile CUDA source with NVRTC, not a default `Nabla::Nabla` package requirement.

- `nbl_target_link_cuda_interop` generates `nbl_cuda_interop_runtime.json` for the target that opted into native CUDA interop.
- The JSON is a build artifact. Nabla packages do not install host-specific CUDA paths.
- Package consumers generate their own JSON when they call `nbl_target_link_cuda_interop`.
- `NBL_CUDA_INTEROP_RUNTIME_JSON` can point runtime discovery at custom JSON files without rebuilding the application.
- Runtime lookup checks explicit JSON paths first, then executable-local JSON, app-local header bundles, explicit include-dir environment variables, `CUDA_PATH` style toolkit roots, Python/conda package layouts, and common system install roots.
- The probe looks for directories that contain CUDA runtime headers. It does not hardcode a CUDA major version in app-local paths.
- `cuda_native::CCUDAHandlerAccessor::compileDirectlyToPTX` appends discovered include directories to NVRTC options. Default discovery is cached after the first call.

Production machines do not need the full CUDA SDK just because Nabla was built with CUDA. Applications that use NVRTC with CUDA runtime headers can provide those headers through generated JSON, a custom JSON path, an app-local bundle, an official runtime/header package, or an installed toolkit.

Nabla does not ship CUDA runtime headers by default. NVIDIA CUDA EULA allows redistribution only for selected components. The distribution section says: "The portions of the SDK that are distributable under the Agreement are listed in Attachment A." Attachment A says: "The following CUDA Toolkit files may be distributed with applications developed by you." See:

- https://docs.nvidia.com/cuda/eula/#distribution
- https://docs.nvidia.com/cuda/eula/#attachment-a

Attachment A lists header groups relevant to NVRTC runtime compilation:

- NVIDIA Runtime Compilation Library and Header: `nvrtc.h`
- CUDA Floating Point Type Headers: `cuda_fp16.h`, `cuda_fp16.hpp`, `cuda_bf16.h`, `cuda_bf16.hpp`, `cuda_fp8.h`, `cuda_fp8.hpp`, `cuda_fp6.h`, `cuda_fp6.hpp`, `cuda_fp4.h`, `cuda_fp4.hpp`
- CUDA Headers for Runtime Compilation: `crt/host_defines.h`, `cuComplex.h`, `cuda_awbarrier_helpers.h`, `cuda_awbarrier_primitives.h`, `cuda_awbarrier.h`, `cuda_pipeline_helpers.h`, `cuda_pipeline_primitives.h`, `cuda_pipeline.h`, `cuda_runtime_api.h`, `cuda.h`, `cuda/std/tuple`, `cuda/std/type_traits`, `cuda/std/utility`, `device_types.h`, `vector_functions.h`, and `vector_types.h`

CuPy documents the same NVRTC issue for CUDA 12.2+. Their install docs say: "On CUDA 12.2 or later, CUDA Runtime header files are required to compile kernels in CuPy." They show the common `vector_types.h` failure and recommend CUDA runtime header packages for PyPI/system package installs:

- https://docs.cupy.dev/en/v13.5.0/install.html#cupy-always-raises-nvrtc-error-compilation-6
- https://github.com/cupy/cupy/issues/8466

## CUDA ON/OFF Builds

- SDK-free public headers stay stable for CUDA ON and CUDA OFF Nabla builds.
- CUDA implementation headers and SDK includes stay behind `_NBL_COMPILE_WITH_CUDA_`.
- CUDA OFF implementations are local stubs in the same `.cpp` files. Factory/import/export paths return `nullptr` for unavailable CUDA features instead of producing unresolved symbols.
- The Nabla source list stays stable, so CUDA interop `.cpp` files remain visible in IDE projects for both CUDA ON and CUDA OFF builds.

## Related Designs

The split follows the same boundary pattern used by mature GPU projects: default headers avoid vendor SDK requirements, native access is explicit, and implementation details stay outside the default public API.

- OpenCV keeps common CUDA-facing headers independent from CUDA Runtime API and exposes raw `cudaStream_t` / `cudaEvent_t` through a separate accessor header: https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/cuda_stream_accessor.hpp#L50-L79
- OpenCV keeps CUDA implementation headers private and includes `cuda.h`, `cuda_runtime.h`, and NPP there: https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/private.cuda.hpp#L47-L61
- Blender/Cycles exposes a CUDA device boundary without CUDA SDK headers in the boundary header: https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device.h#L7-L27
- Blender/Cycles keeps `CUdevice`, `CUcontext`, `cuda.h`, and `cuew.h` in the CUDA implementation header/source: https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device_impl.h#L12-L30
