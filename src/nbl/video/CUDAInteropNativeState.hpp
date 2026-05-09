#ifndef _NBL_VIDEO_CUDA_INTEROP_NATIVE_STATE_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_NATIVE_STATE_H_INCLUDED_

#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"

#include <array>

namespace nbl::video
{

struct CCUDAHandler::SNativeState
{
	struct SDeviceState
	{
		cuda_native::SCUDADeviceInfo info = {};
		std::array<int,CU_DEVICE_ATTRIBUTE_MAX> attributes = {};
	};

	cuda_native::CUDA cuda;
	cuda_native::NVRTC nvrtc;
	core::vector<cuda_native::SCUDADeviceInfo> availableDevices;
	core::vector<SDeviceState> deviceStates;

	SNativeState(cuda_native::CUDA&& _cuda, cuda_native::NVRTC&& _nvrtc)
		: cuda(std::move(_cuda))
		, nvrtc(std::move(_nvrtc))
	{}
};

struct CCUDADevice::SNativeState
{
	CUdevice handle = {};
	CUcontext context = nullptr;

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

struct SAccess
{
	static CCUDAHandler::SNativeState& native(CCUDAHandler& handler) { return *handler.m_native; }
	static const CCUDAHandler::SNativeState& native(const CCUDAHandler& handler) { return *handler.m_native; }

	static CCUDADevice::SNativeState& native(CCUDADevice& device) { return *device.m_native; }
	static const CCUDADevice::SNativeState& native(const CCUDADevice& device) { return *device.m_native; }

	static CCUDAExportableMemory::SNativeState& native(CCUDAExportableMemory& memory) { return *memory.m_native; }
	static const CCUDAExportableMemory::SNativeState& native(const CCUDAExportableMemory& memory) { return *memory.m_native; }
	static std::unique_ptr<CCUDAExportableMemory::SNativeState> makeExportableMemoryNativeState()
	{
		return std::unique_ptr<CCUDAExportableMemory::SNativeState>(new CCUDAExportableMemory::SNativeState());
	}
	static CUdeviceptr& deviceptr(CCUDAExportableMemory::SNativeState& nativeState) { return nativeState.ptr; }
	static core::smart_refctd_ptr<CCUDAExportableMemory> makeExportableMemory(core::smart_refctd_ptr<CCUDADevice> device, CCUDAExportableMemory::SCachedCreationParams&& params, std::unique_ptr<CCUDAExportableMemory::SNativeState>&& nativeState)
	{
		return CCUDAExportableMemory::create(std::move(device),std::move(params),std::move(nativeState));
	}

	static CCUDAImportedMemory::SNativeState& native(CCUDAImportedMemory& memory) { return *memory.m_native; }
	static const CCUDAImportedMemory::SNativeState& native(const CCUDAImportedMemory& memory) { return *memory.m_native; }

	static CCUDAImportedSemaphore::SNativeState& native(CCUDAImportedSemaphore& semaphore) { return *semaphore.m_native; }
	static const CCUDAImportedSemaphore::SNativeState& native(const CCUDAImportedSemaphore& semaphore) { return *semaphore.m_native; }

	static system::logger_opt_ptr logger(const CCUDAHandler& handler) { return handler.m_logger.get().get(); }
	static system::logger_opt_ptr logger(const CCUDADevice& device) { return device.m_logger; }
	static const CCUDADevice* device(const CCUDAImportedMemory& memory) { return memory.m_device.get(); }
	static IDeviceMemoryAllocation* source(const CCUDAImportedMemory& memory) { return memory.m_src.get(); }
	static CUmemAllocationHandleType allocationHandleType()
	{
	#ifdef _WIN32
		return CU_MEM_HANDLE_TYPE_WIN32;
	#else
		return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	#endif
	}
};

}

}

#endif
