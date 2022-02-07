// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CCUDAHandler.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "jitify/jitify.hpp"

namespace nbl
{
namespace cuda
{
CCUDAHandler::CUDA CCUDAHandler::cuda;
CCUDAHandler::NVRTC CCUDAHandler::nvrtc;

core::vector<CCUDAHandler::Device> CCUDAHandler::devices;

core::vector<core::smart_refctd_ptr<const io::IReadFile> > CCUDAHandler::headers;
core::vector<const char*> CCUDAHandler::headerContents;
core::vector<const char*> CCUDAHandler::headerNames;

CUresult CCUDAHandler::init()
{
    if(CudaVersion)
        return CUDA_SUCCESS;

    CUresult result = CUDA_ERROR_UNKNOWN;
    auto cleanup = core::makeRAIIExiter([&result]() -> void {
        if(result != CUDA_SUCCESS)
        {
            CudaVersion = 0;
            DeviceCount = 0;
            devices.clear();
        }
    });

    cuda = CUDA(
#if defined(_NBL_WINDOWS_API_)
        "nvcuda"
#elif defined(_NBL_POSIX_API_)
        "cuda"
#endif
    );
#define SAFE_CUDA_CALL(NO_PTR_ERROR, FUNC, ...) \
    {                                           \
        if(!cuda.p##FUNC)                       \
            return result = NO_PTR_ERROR;       \
        result = cuda.p##FUNC##(__VA_ARGS__);   \
        if(result != CUDA_SUCCESS)              \
            return result;                      \
    }

    SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED, cuInit, 0)

    SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED, cuDriverGetVersion, &CudaVersion)
    if(CudaVersion < 9000)
        return result = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;

    SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED, cuDeviceGetCount, &DeviceCount)

    devices.resize(DeviceCount);
    int j = 0;
    for(int i = 0; i < DeviceCount; i++)
    {
        auto tmp = Device(i);

        const int archVersion[2] = {tmp.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR], tmp.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR]};
        if(archVersion[0] > 8 || archVersion[0] == 8 && archVersion[1] > 0)
        {
            assert(strcmp(virtualCUDAArchitectures[13], "-arch=compute_80") == 0);
            if(!virtualCUDAArchitecture)
                virtualCUDAArchitecture = virtualCUDAArchitectures[13];
        }
        else
        {
            const std::string virtualArchString = "-arch=compute_" + std::to_string(archVersion[0]) + std::to_string(archVersion[1]);

            int32_t i = sizeof(virtualCUDAArchitectures) / sizeof(const char*);
            while(i > 0)
                if(virtualCUDAArchitecture == virtualCUDAArchitectures[--i] || !virtualCUDAArchitecture)
                    break;

            if(!virtualCUDAArchitecture || virtualArchString != virtualCUDAArchitecture)
            {
                i++;
                while(i > 0)
                    if(virtualArchString == virtualCUDAArchitectures[--i])
                    {
                        virtualCUDAArchitecture = virtualCUDAArchitectures[i];
                        break;
                    }
            }
        }

        devices[j++] = tmp;
    }
    devices.resize(j);

    if(!virtualCUDAArchitecture)
        return result = CUDA_ERROR_INVALID_DEVICE;

#undef SAFE_CUDA_CALL

#if defined(_NBL_WINDOWS_API_)
    const char* nvrtc64_versions[] = {"nvrtc64_111", "nvrtc64_110", "nvrtc64_102", "nvrtc64_101", "nvrtc64_100", "nvrtc64_92", "nvrtc64_91", "nvrtc64_90", "nvrtc64_80", "nvrtc64_75", "nvrtc64_70", nullptr};
    const char* nvrtc64_suffices[] = {"", "_", "_0", "_1", "_2", nullptr};
    for(auto verpath = nvrtc64_versions; *verpath; verpath++)
    {
        for(auto suffix = nvrtc64_suffices; *suffix; suffix++)
        {
            std::string path(*verpath);
            path += *suffix;
            nvrtc = NVRTC(path.c_str());
            if(nvrtc.pnvrtcVersion)
                break;
        }
        if(nvrtc.pnvrtcVersion)
            break;
    }
#elif defined(_NBL_POSIX_API_)
    nvrtc = NVRTC("nvrtc");
//nvrtc_builtins = NVRTC("nvrtc-builtins");
#endif

    int nvrtcVersion[2] = {-1, -1};
    cuda::CCUDAHandler::nvrtc.pnvrtcVersion(nvrtcVersion + 0, nvrtcVersion + 1);
    if(nvrtcVersion[0] < 9)
        return result = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;

    for(const auto& it : jitify::detail::get_jitsafe_headers_map())
    {
        auto file = core::make_smart_refctd_ptr<io::CMemoryReadFile>(it.second.c_str(), it.second.size() + 1ull, it.first.c_str());
        headerContents.push_back(reinterpret_cast<const char*>(file->getData()));
        headerNames.push_back(file->getFileName().c_str());
        headers.push_back(core::move_and_static_cast<const io::IReadFile>(file));
    }

    return result = CUDA_SUCCESS;
}

void CCUDAHandler::deinit()
{
    CudaVersion = 0;
    DeviceCount = 0;
    devices.resize(0u);

    cuda = CUDA();

    nvrtc = NVRTC();

    headerContents.clear();
    headerNames.clear();
    headers.clear();
}

CCUDAHandler::Device::Device(int ordinal)
    : Device()
{
    if(cuda.pcuDeviceGet(&handle, ordinal) != CUDA_SUCCESS)
    {
        handle = 0;
        return;
    }

    uint32_t namelen = sizeof(name);
    if(cuda.pcuDeviceGetName(name, namelen, handle) != CUDA_SUCCESS)
        return;

    if(cuda.pcuDeviceGetLuid(&luid, &deviceNodeMask, handle))
        return;

    if(cuda.pcuDeviceGetUuid(&uuid, handle) != CUDA_SUCCESS)
        return;

    if(cuda.pcuDeviceTotalMem_v2(&vram_size, handle) != CUDA_SUCCESS)
        return;

    for(int i = 0; i < CU_DEVICE_ATTRIBUTE_MAX; i++)
        cuda.pcuDeviceGetAttribute(attributes + i, static_cast<CUdevice_attribute>(i), handle);
}

CUresult CCUDAHandler::registerBuffer(GraphicsAPIObjLink<video::IGPUBuffer>* link, uint32_t flags)
{
    assert(link->obj);
    auto glbuf = static_cast<video::COpenGLBuffer*>(link->obj.get());
    auto retval = cuda.pcuGraphicsGLRegisterBuffer(&link->cudaHandle, glbuf->getOpenGLName(), flags);
    if(retval != CUDA_SUCCESS)
        link->obj = nullptr;
    return retval;
}
CUresult CCUDAHandler::registerImage(GraphicsAPIObjLink<video::IGPUImage>* link, uint32_t flags)
{
    assert(link->obj);

    auto format = link->obj->getCreationParameters().format;
    if(asset::isBlockCompressionFormat(format) || asset::isDepthOrStencilFormat(format) || asset::isScaledFormat(format) || asset::isPlanarFormat(format))
        return CUDA_ERROR_INVALID_IMAGE;

    auto glimg = static_cast<video::COpenGLImage*>(link->obj.get());
    GLenum target = glimg->getOpenGLTarget();
    switch(target)
    {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_3D:
            break;
        default:
            return CUDA_ERROR_INVALID_IMAGE;
            break;
    }
    auto retval = cuda.pcuGraphicsGLRegisterImage(&link->cudaHandle, glimg->getOpenGLName(), target, flags);
    if(retval != CUDA_SUCCESS)
        link->obj = nullptr;
    return retval;
}

constexpr auto MaxAquireOps = 4096u;

CUresult CCUDAHandler::acquireAndGetPointers(GraphicsAPIObjLink<video::IGPUBuffer>* linksBegin, GraphicsAPIObjLink<video::IGPUBuffer>* linksEnd, CUstream stream, size_t* outbufferSizes)
{
    if(linksBegin + MaxAquireOps < linksEnd)
        return CUDA_ERROR_OUT_OF_MEMORY;
    alignas(_NBL_SIMD_ALIGNMENT) uint8_t stackScratch[MaxAquireOps * sizeof(void*)];

    CUresult result = acquireResourcesFromGraphics(stackScratch, linksBegin, linksEnd, stream);
    if(result != CUDA_SUCCESS)
        return result;

    size_t tmp = 0xdeadbeefbadc0ffeull;
    size_t* sit = outbufferSizes;
    for(auto iit = linksBegin; iit != linksEnd; iit++, sit++)
    {
        if(!iit->acquired)
            return CUDA_ERROR_UNKNOWN;

        result = cuda::CCUDAHandler::cuda.pcuGraphicsResourceGetMappedPointer_v2(&iit->asBuffer.pointer, outbufferSizes ? sit : &tmp, iit->cudaHandle);
        if(result != CUDA_SUCCESS)
            return result;
    }
    return CUDA_SUCCESS;
}
CUresult CCUDAHandler::acquireAndGetMipmappedArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, CUstream stream)
{
    if(linksBegin + MaxAquireOps < linksEnd)
        return CUDA_ERROR_OUT_OF_MEMORY;
    alignas(_NBL_SIMD_ALIGNMENT) uint8_t stackScratch[MaxAquireOps * sizeof(void*)];

    CUresult result = acquireResourcesFromGraphics(stackScratch, linksBegin, linksEnd, stream);
    if(result != CUDA_SUCCESS)
        return result;

    for(auto iit = linksBegin; iit != linksEnd; iit++)
    {
        if(!iit->acquired)
            return CUDA_ERROR_UNKNOWN;

        result = cuda::CCUDAHandler::cuda.pcuGraphicsResourceGetMappedMipmappedArray(&iit->asImage.mipmappedArray, iit->cudaHandle);
        if(result != CUDA_SUCCESS)
            return result;
    }
    return CUDA_SUCCESS;
}
CUresult CCUDAHandler::acquireAndGetArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, uint32_t* arrayIndices, uint32_t* mipLevels, CUstream stream)
{
    if(linksBegin + MaxAquireOps < linksEnd)
        return CUDA_ERROR_OUT_OF_MEMORY;
    alignas(_NBL_SIMD_ALIGNMENT) uint8_t stackScratch[MaxAquireOps * sizeof(void*)];

    CUresult result = acquireResourcesFromGraphics(stackScratch, linksBegin, linksEnd, stream);
    if(result != CUDA_SUCCESS)
        return result;

    auto ait = arrayIndices;
    auto mit = mipLevels;
    for(auto iit = linksBegin; iit != linksEnd; iit++, ait++, mit++)
    {
        if(!iit->acquired)
            return CUDA_ERROR_UNKNOWN;

        result = cuda::CCUDAHandler::cuda.pcuGraphicsSubResourceGetMappedArray(&iit->asImage.array, iit->cudaHandle, *ait, *mit);
        if(result != CUDA_SUCCESS)
            return result;
    }
    return CUDA_SUCCESS;
}

bool CCUDAHandler::defaultHandleResult(CUresult result)
{
    switch(result)
    {
        case CUDA_SUCCESS:
            return true;
            break;
        case CUDA_ERROR_INVALID_VALUE:
            printf(R"===(CCUDAHandler:
				This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
			)===");
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            printf(R"===(CCUDAHandler:
				The API call failed because it was unable to allocate enough memory to perform the requested operation.
			)===");
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            printf(R"===(CCUDAHandler:
				This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed. 
			)===");
            break;
        case CUDA_ERROR_DEINITIALIZED:
            printf(R"===(CCUDAHandler:
				This indicates that the CUDA driver is in the process of shutting down.
			)===");
            break;
        case CUDA_ERROR_PROFILER_DISABLED:
            printf(R"===(CCUDAHandler:
				This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
			)===");
            break;
        case CUDA_ERROR_NO_DEVICE:
            printf(R"===(CCUDAHandler:
				This indicates that no CUDA-capable devices were detected by the installed CUDA driver. 
			)===");
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            printf(R"===(CCUDAHandler:
				This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device. 
			)===");
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            printf(R"===(CCUDAHandler:
				This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module. 
			)===");
            break;
        case CUDA_ERROR_INVALID_CONTEXT:
            printf(R"===(CCUDAHandler:
				This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
			)===");
            break;
        case CUDA_ERROR_MAP_FAILED:
            printf(R"===(CCUDAHandler:
				This indicates that a map or register operation has failed.
			)===");
            break;
        case CUDA_ERROR_UNMAP_FAILED:
            printf(R"===(CCUDAHandler:
				This indicates that an unmap or unregister operation has failed.
			)===");
            break;
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            printf(R"===(CCUDAHandler:
				This indicates that the specified array is currently mapped and thus cannot be destroyed.
			)===");
            break;
        case CUDA_ERROR_ALREADY_MAPPED:
            printf(R"===(CCUDAHandler:
				This indicates that the resource is already mapped.
			)===");
            break;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            printf(R"===(CCUDAHandler:
				This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
			)===");
            break;
        case CUDA_ERROR_ALREADY_ACQUIRED:
            printf(R"===(CCUDAHandler:
				This indicates that a resource has already been acquired. 
			)===");
            break;
        case CUDA_ERROR_NOT_MAPPED:
            printf(R"===(CCUDAHandler:
				This indicates that a resource is not mapped.
			)===");
            break;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            printf(R"===(CCUDAHandler:
				This indicates that a mapped resource is not available for access as an array. 
			)===");
            break;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            printf(R"===(CCUDAHandler:
				This indicates that a mapped resource is not available for access as a pointer. 
			)===");
            break;
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            printf(R"===(CCUDAHandler:
				This indicates that an uncorrectable ECC error was detected during execution. 
			)===");
            break;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            printf(R"===(CCUDAHandler:
				This indicates that the CUlimit passed to the API call is not supported by the active device. 
			)===");
            break;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            printf(R"===(CCUDAHandler:
				This indicates that the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread. 
			)===");
            break;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            printf(R"===(CCUDAHandler:
				This indicates that peer access is not supported across the given devices. 
			)===");
            break;
        case CUDA_ERROR_INVALID_PTX:
            printf(R"===(CCUDAHandler:
				This indicates that a PTX JIT compilation failed. 
			)===");
            break;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            printf(R"===(CCUDAHandler:
				This indicates an error with OpenGL or DirectX context. 
			)===");
            break;
        case CUDA_ERROR_NVLINK_UNCORRECTABLE:
            printf(R"===(CCUDAHandler:
				This indicates that an uncorrectable NVLink error was detected during the execution. 
			)===");
            break;
        case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
            printf(R"===(CCUDAHandler:
				This indicates that the PTX JIT compiler library was not found. 
			)===");
            break;
        case CUDA_ERROR_INVALID_SOURCE:
            printf(R"===(CCUDAHandler:
				This indicates that the device kernel source is invalid. 
			)===");
            break;
        case CUDA_ERROR_FILE_NOT_FOUND:
            printf(R"===(CCUDAHandler:
				This indicates that the file specified was not found. 
			)===");
            break;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            printf(R"===(CCUDAHandler:
				This indicates that a link to a shared object failed to resolve.
			)===");
            break;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            printf(R"===(CCUDAHandler:
				This indicates that initialization of a shared object failed.
			)===");
            break;
        case CUDA_ERROR_OPERATING_SYSTEM:
            printf(R"===(CCUDAHandler:
				This indicates that an OS call failed. 
			)===");
            break;
        case CUDA_ERROR_INVALID_HANDLE:
            printf(R"===(CCUDAHandler:
				This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent. 
			)===");
            break;
        case CUDA_ERROR_ILLEGAL_STATE:
            printf(R"===(CCUDAHandler:
				This indicates that a resource required by the API call is not in a valid state to perform the requested operation. 
			)===");
            break;
        case CUDA_ERROR_NOT_FOUND:
            printf(R"===(CCUDAHandler:
				This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names. 
			)===");
            break;
        case CUDA_ERROR_NOT_READY:
            printf(R"===(CCUDAHandler:
				This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().
			)===");
            break;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            printf(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===");
            break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            printf(R"===(CCUDAHandler:
				This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error. 
			)===");
            break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            printf(R"===(CCUDAHandler:
				This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===");
            break;
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            printf(R"===(CCUDAHandler:
				This error indicates a kernel launch that uses an incompatible texturing mode. 
			)===");
            break;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            printf(R"===(CCUDAHandler:
				This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled. 
			)===");
            break;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            printf(R"===(CCUDAHandler:
				This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess(). 
			)===");
            break;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            printf(R"===(CCUDAHandler:
				This error indicates that the primary context for the specified device has already been initialized. 
			)===");
            break;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            printf(R"===(CCUDAHandler:
				This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized. 
			)===");
            break;
        case CUDA_ERROR_ASSERT:
            printf(R"===(CCUDAHandler:
				A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA. 
			)===");
            break;
        case CUDA_ERROR_TOO_MANY_PEERS:
            printf(R"===(CCUDAHandler:
				This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess(). 
			)===");
            break;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            printf(R"===(CCUDAHandler:
				This error indicates that the memory range passed to cuMemHostRegister() has already been registered. 
			)===");
            break;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            printf(R"===(CCUDAHandler:
				This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region. 
			)===");
            break;
        case CUDA_ERROR_HARDWARE_STACK_ERROR:
            printf(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===");
            break;
        case CUDA_ERROR_ILLEGAL_INSTRUCTION:
            printf(R"===(CCUDAHandler:
				While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===");
            break;
        case CUDA_ERROR_MISALIGNED_ADDRESS:
            printf(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===");
            break;
        case CUDA_ERROR_INVALID_ADDRESS_SPACE:
            printf(R"===(CCUDAHandler:
				While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===");
            break;
        case CUDA_ERROR_INVALID_PC:
            printf(R"===(CCUDAHandler:
				While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===");
            break;
        case CUDA_ERROR_LAUNCH_FAILED:
            printf(R"===(CCUDAHandler:
				An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===");
            break;
        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            printf(R"===(CCUDAHandler:
				This error indicates that the number of blocks launched per grid for a kernel that was launched via either cuLaunchCooperativeKernel or cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor or cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
			)===");
            break;
        case CUDA_ERROR_NOT_PERMITTED:
            printf(R"===(CCUDAHandler:
				This error indicates that the attempted operation is not permitted.
			)===");
            break;
        case CUDA_ERROR_NOT_SUPPORTED:
            printf(R"===(CCUDAHandler:
				This error indicates that the attempted operation is not supported on the current system or device.
			)===");
            break;
        case CUDA_ERROR_SYSTEM_NOT_READY:
            printf(R"===(CCUDAHandler:
				This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
			)===");
            break;
        case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
            printf(R"===(CCUDAHandler:
				This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions. 
			)===");
            break;
        case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
            printf(R"===(CCUDAHandler:
				This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
            printf(R"===(CCUDAHandler:
				This error indicates that the operation is not permitted when the stream is capturing. 
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
            printf(R"===(CCUDAHandler:
				This error indicates that the current capture sequence on the stream has been invalidated due to a previous error. 
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_MERGE:
            printf(R"===(CCUDAHandler:
				This error indicates that the operation would have resulted in a merge of two independent capture sequences. 
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
            printf(R"===(CCUDAHandler:
				This error indicates that the capture was not initiated in this stream. 
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
            printf(R"===(CCUDAHandler:
				This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
            printf(R"===(CCUDAHandler:
				This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
            printf(R"===(CCUDAHandler:
				This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy. 
			)===");
            break;
        case CUDA_ERROR_CAPTURED_EVENT:
            printf(R"===(CCUDAHandler:
				This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream. 
			)===");
            break;
        case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
            printf(R"===(CCUDAHandler:
				A stream capture sequence not initiated with the CU_STREAM_CAPTURE_MODE_RELAXED argument to cuStreamBeginCapture was passed to cuStreamEndCapture in a different thread. 
			)===");
            break;
        case CUDA_ERROR_TIMEOUT:
            printf(R"===(CCUDAHandler:
				This error indicates that the timeout specified for the wait operation has lapsed. 
			)===");
            break;
        case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
            printf(R"===(CCUDAHandler:
				This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update. 
			)===");
            break;
        case CUDA_ERROR_UNKNOWN:
        default:
            printf("CCUDAHandler: Unknown CUDA Error!\n");
            break;
    }
    _NBL_DEBUG_BREAK_IF(true);
    return false;
}

nvrtcResult CCUDAHandler::getProgramLog(nvrtcProgram prog, std::string& log)
{
    size_t _size = 0ull;
    nvrtcResult sizeRes = nvrtc.pnvrtcGetProgramLogSize(prog, &_size);
    if(sizeRes != NVRTC_SUCCESS)
        return sizeRes;
    if(_size == 0ull)
        return NVRTC_ERROR_INVALID_INPUT;

    log.resize(_size);
    return nvrtc.pnvrtcGetProgramLog(prog, log.data());
}

nvrtcResult CCUDAHandler::getPTX(nvrtcProgram prog, std::string& ptx)
{
    size_t _size = 0ull;
    nvrtcResult sizeRes = nvrtc.pnvrtcGetPTXSize(prog, &_size);
    if(sizeRes != NVRTC_SUCCESS)
        return sizeRes;
    if(_size == 0ull)
        return NVRTC_ERROR_INVALID_INPUT;

    ptx.resize(_size);
    return nvrtc.pnvrtcGetPTX(prog, ptx.data());
}

}
}

#endif  // _NBL_COMPILE_WITH_CUDA_
