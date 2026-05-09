# CUDA Interop

## Layout

- `Nabla::Nabla` owns the SDK-free CUDA interop API in `nbl/video/CCUDA*.h` and the implementation in `src/nbl/video/CCUDA*.cpp`.
- The public Nabla headers do not include `cuda.h`, `nvrtc.h`, or other CUDA SDK headers. A consumer that only links `Nabla::Nabla` does not need a CUDA SDK install just to parse Nabla headers.
- CUDA native state is stored behind incomplete `SNativeState` members in Nabla classes. Public headers expose fixed-layout opaque value handles from `nbl/video/CUDAInteropHandles.h`.
- `Nabla::ext::CUDAInterop` is an `INTERFACE` target. It builds no artifact. It only adds the SDK opt-in header, `CUDA::toolkit`, and runtime-header discovery setup to targets that ask for raw CUDA interop.
- `CUDAInteropNative.h` is the only opt-in header that includes CUDA SDK headers. It maps Nabla opaque handles to CUDA SDK types with `cuda_native::SNativeHandle`.

## CMake Usage

`Nabla::Nabla`-only usage stays SDK-free:

```cmake
find_package(Nabla CONFIG REQUIRED)
target_link_libraries(app PRIVATE Nabla::Nabla)
```

SDK-typed CUDA interop is explicit:

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

Consumers can also choose the SDK used for SDK-typed compilation with:

```cmake
cmake -S . -B build -DNabla_CUDA_TOOLKIT_ROOT=<cuda-root>
```

This affects SDK opt-in compilation and generated runtime header discovery only. It does not rebuild Nabla and does not change the `Nabla.dll` ABI.

## SDK Opt-In Usage

```cpp
#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"

auto handler = nbl::video::CCUDAHandler::create(system, std::move(logger));
auto cudaDevice = handler->createDevice(std::move(vulkanConnection), physicalDevice);

if (!nbl::video::cuda_native::isBuildCUDAVersionCompatible())
    return false;

auto memory = cudaDevice->createExportableMemory({
    .size = size,
    .alignment = alignment,
    .locationType = CU_MEM_LOCATION_TYPE_DEVICE,
});

nbl::video::cuda_native::SCUdeviceptr mapped;
if (importedMemory)
    importedMemory->getMappedBuffer(mapped.opaque());

CUdeviceptr rawMapped = mapped;
CUdeviceptr rawExported = nbl::video::cuda_native::SCUdeviceptr(memory->getDeviceptr());
auto& cu = handler->getCUDAFunctionTable();
auto& nvrtc = handler->getNVRTCFunctionTable();

std::string log;
auto compile = nbl::video::cuda_native::compileDirectlyToPTX(
    *handler,
    std::move(cudaSource),
    "kernel.cu",
    cudaDevice->geDefaultCompileOptions(),
    log
);
```

SDK opt-in access is not a full CUDA wrapper. It is the glue between Nabla resource lifetime and raw CUDA interop:

- `CCUDAHandler::getCUDAFunctionTable` and `CCUDAHandler::getNVRTCFunctionTable` expose the loaded Driver API and NVRTC tables after SDK opt-in.
- `cuda_native::SNativeHandle<T>` converts between SDK-free Nabla opaque handles and CUDA SDK handles such as `CUdeviceptr`.
- CUDA enum values can be passed to SDK-free Nabla methods such as `CCUDADevice::createExportableMemory` and `CCUDADevice::roundToGranularity`. Nabla stores them as integer values in its public ABI.
- `CCUDAImportedMemory::getMappedBuffer` writes an opaque `cuda_interop::SCUdeviceptr`. SDK opt-in code can pass `cuda_native::SCUdeviceptr::opaque()` and then use the wrapper as `CUdeviceptr`.
- `compileDirectlyToPTX` returns PTX/result and writes the NVRTC log to a required `std::string&`.

Smoke examples:

- `src/nbl/ext/CUDAInterop/smoke/public_boundary.cpp` checks that `Nabla::Nabla` headers stay SDK-free.
- `src/nbl/ext/CUDAInterop/smoke/clean_opt_in.cpp` checks `Nabla::Nabla` package usage without SDK opt-in.
- `src/nbl/ext/CUDAInterop/smoke/native_opt_in.cpp` checks SDK opt-in, runtime header discovery, `cuda_fp16.h`, NVRTC, and raw interop usage.

## ABI

- `CCUDAHandler`, `CCUDADevice`, `CCUDAExportableMemory`, `CCUDAImportedMemory`, and `CCUDAImportedSemaphore` are exported from `Nabla.dll` through the normal Nabla ABI.
- Their public declarations do not expose CUDA SDK structs, CUDA SDK layouts, or `cuda.h` / `nvrtc.h` includes.
- Opaque handle types are small trivially-copyable byte arrays with fixed size/alignment chosen to match CUDA SDK handle storage. The SDK opt-in header validates this with `static_assert`s against the SDK used by the consumer.
- CUDA implementation state is owned by Nabla through private `SNativeState` members. Consumers cannot construct CUDA wrapper objects with arbitrary internal CUDA state.
- SDK-sized arrays, CUDA enum storage, and CUDA implementation headers stay private to Nabla.
- A consumer can build SDK opt-in code with its own compatible SDK independently from the SDK used to build Nabla. SDK-typed code can check `cuda_native::isBuildCUDAVersionCompatible()` when exact CUDA SDK version matching is required.
- Runtime include-option construction is header-only and is not part of the exported ABI.
- The loaded CUDA driver and NVRTC runtime are validated at runtime.

## Runtime Header Discovery

NVRTC may need CUDA runtime headers when user kernels include files such as `cuda_fp16.h`, `vector_types.h`, or `cuda_runtime_api.h`. This is a runtime concern of applications that compile CUDA source with NVRTC, not a `Nabla::Nabla` package requirement.

- `nbl_target_link_cuda_interop` generates `nbl_cuda_interop_runtime.json` for the target that opted into SDK-typed CUDA interop.
- The JSON is a build artifact. Nabla packages do not install host-specific CUDA paths.
- Package consumers generate their own JSON when they call `nbl_target_link_cuda_interop`.
- `NBL_CUDA_INTEROP_RUNTIME_JSON` can point runtime discovery at custom JSON files without rebuilding the application.
- Runtime lookup checks explicit JSON paths first, then executable-local JSON, app-local header bundles, explicit include-dir environment variables, `CUDA_PATH` style toolkit roots, Python/conda package layouts, and common system install roots.
- The probe looks for directories that contain CUDA runtime headers. It does not hardcode a CUDA major version in app-local paths.
- `cuda_native::compileDirectlyToPTX` appends discovered include directories to NVRTC options. Discovery is cached after the first call.

Production machines do not need the full CUDA SDK just because Nabla was built with CUDA. Applications that use NVRTC with CUDA runtime headers can provide those headers through generated JSON, a custom JSON path, an app-local bundle, an official runtime/header package, or an installed toolkit.

Nabla could ship an app-local bundle of selected CUDA runtime headers and make it available to runtime discovery. That model is allowed by the NVIDIA CUDA EULA for the components listed in Attachment A. Nabla intentionally does not bundle these headers. Because of that, end users should prefer an official CUDA runtime/header package for production machines. An installed toolkit also works, but the full toolkit is mainly for developers compiling Nabla or SDK-typed CUDA code.

NVIDIA CUDA EULA allows redistribution only for selected components. The distribution section says: "The portions of the SDK that are distributable under the Agreement are listed in Attachment A." Attachment A says: "The following CUDA Toolkit files may be distributed with applications developed by you." See:

- https://docs.nvidia.com/cuda/eula/#distribution
- https://docs.nvidia.com/cuda/eula/#attachment-a

This means the Attachment A header groups below can be redistributed with applications under the EULA terms. It does not mean the full CUDA SDK can be redistributed. Applications that need NVRTC runtime compilation can decide whether to ship the allowed headers, depend on an official runtime/header package, or point discovery at an installed toolkit/header package.

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

The split follows the same boundary pattern used by mature GPU projects: public/common headers avoid vendor SDK requirements, vendor SDK access is explicit, and implementation details stay outside the public API.

- OpenCV keeps common CUDA-facing headers independent from CUDA Runtime API and exposes raw `cudaStream_t` / `cudaEvent_t` through a separate accessor header: https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/cuda_stream_accessor.hpp#L50-L79
- OpenCV keeps CUDA implementation headers private and includes `cuda.h`, `cuda_runtime.h`, and NPP there: https://github.com/opencv/opencv/blob/808d2d596c475d95fedb6025c9ed425d62bba04c/modules/core/include/opencv2/core/private.cuda.hpp#L47-L61
- Blender/Cycles exposes a CUDA device boundary without CUDA SDK headers in the boundary header: https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device.h#L7-L27
- Blender/Cycles keeps `CUdevice`, `CUcontext`, `cuda.h`, and `cuew.h` in the CUDA implementation header/source: https://github.com/blender/blender/blob/794c527e8595a9f448e0143a217d0ceb648c5e7e/intern/cycles/device/cuda/device_impl.h#L12-L30
- OpenMM keeps the CUDA platform boundary on OpenMM types/properties in `CudaPlatform.h`, while `CudaContext.h` is the CUDA-specific low-level header that includes CUDA SDK headers and exposes `CUmodule` / `CUfunction`: https://github.com/openmm/openmm/blob/master/platforms/cuda/include/CudaPlatform.h#L48-L120 and https://github.com/openmm/openmm/blob/master/platforms/cuda/include/CudaContext.h#L32-L52
- GROMACS gates CUDA source handling behind `GMX_GPU_CUDA` in the library build and keeps CUDA runtime types in internal GPU utility headers: https://gitlab.com/gromacs/gromacs/-/blob/main/src/gromacs/CMakeLists.txt#L339-L367 and https://gitlab.com/gromacs/gromacs/-/blob/main/src/gromacs/gpu_utils/gputraits.cuh#L44-L58
- ONNX Runtime keeps the public C API provider-neutral and routes CUDA through provider-specific bridge/factory code: https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h#L1-L80 and https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/provider_bridge_ort.cc#L110-L150
