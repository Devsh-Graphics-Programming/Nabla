#ifndef _NBL_EXT_CUDA_INTEROP_NATIVE_STATE_H_INCLUDED_
#define _NBL_EXT_CUDA_INTEROP_NATIVE_STATE_H_INCLUDED_

#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"

#include <array>

namespace nbl::video
{

struct CCUDAHandler::SNativeState
{
	cuda_native::CUDA cuda;
	cuda_native::NVRTC nvrtc;
	core::vector<cuda_native::SCUDADeviceInfo> availableDevices;

	SNativeState(cuda_native::CUDA&& _cuda, cuda_native::NVRTC&& _nvrtc)
		: cuda(std::move(_cuda))
		, nvrtc(std::move(_nvrtc))
	{}
};

struct CCUDADevice::SNativeState
{
	CUdevice handle = {};
	CUcontext context = nullptr;
	std::array<size_t,5> allocationGranularity = {};

	explicit SNativeState(CUdevice _handle)
		: handle(_handle)
	{}
};

struct CCUDAExportableMemory::SNativeState
{
	CUdeviceptr ptr = 0;
};

struct CCUDAImportedMemory::SNativeState
{
	CUexternalMemory handle = nullptr;

	explicit SNativeState(CUexternalMemory _handle)
		: handle(_handle)
	{}
};

struct CCUDAImportedSemaphore::SNativeState
{
	CUexternalSemaphore handle = nullptr;

	explicit SNativeState(CUexternalSemaphore _handle)
		: handle(_handle)
	{}
};

namespace cuda_native
{

inline CUmemLocationType toNative(ECUDAMemoryLocation location)
{
	return static_cast<CUmemLocationType>(static_cast<uint32_t>(location));
}

inline ECUDAMemoryLocation toNabla(CUmemLocationType location)
{
	return static_cast<ECUDAMemoryLocation>(static_cast<uint32_t>(location));
}

inline CUmemAllocationHandleType getAllocationHandleType()
{
#ifdef _WIN32
	return CU_MEM_HANDLE_TYPE_WIN32;
#else
	return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
}

struct SAccess
{
	static CCUDAHandler::SNativeState& native(CCUDAHandler& handler) { return *handler.m_native; }
	static const CCUDAHandler::SNativeState& native(const CCUDAHandler& handler) { return *handler.m_native; }

	static CCUDADevice::SNativeState& native(CCUDADevice& device) { return *device.m_native; }
	static const CCUDADevice::SNativeState& native(const CCUDADevice& device) { return *device.m_native; }

	static CCUDAExportableMemory::SNativeState& native(CCUDAExportableMemory& memory) { return *memory.m_native; }
	static const CCUDAExportableMemory::SNativeState& native(const CCUDAExportableMemory& memory) { return *memory.m_native; }

	static CCUDAImportedMemory::SNativeState& native(CCUDAImportedMemory& memory) { return *memory.m_native; }
	static const CCUDAImportedMemory::SNativeState& native(const CCUDAImportedMemory& memory) { return *memory.m_native; }

	static CCUDAImportedSemaphore::SNativeState& native(CCUDAImportedSemaphore& semaphore) { return *semaphore.m_native; }
	static const CCUDAImportedSemaphore::SNativeState& native(const CCUDAImportedSemaphore& semaphore) { return *semaphore.m_native; }

	static system::logger_opt_ptr logger(const CCUDAHandler& handler) { return handler.m_logger.get().get(); }
	static system::logger_opt_ptr logger(const CCUDADevice& device) { return device.m_logger; }
	static const CCUDADevice* device(const CCUDAImportedMemory& memory) { return memory.m_device.get(); }
	static IDeviceMemoryAllocation* source(const CCUDAImportedMemory& memory) { return memory.m_src.get(); }
};

}

}

#endif
