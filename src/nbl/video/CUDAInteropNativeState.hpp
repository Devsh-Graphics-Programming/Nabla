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

}

#endif
