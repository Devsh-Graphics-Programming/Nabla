#ifndef _NBL_VIDEO_CUDA_INTEROP_NATIVE_STATE_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_NATIVE_STATE_H_INCLUDED_

#include "nbl/video/CUDAInteropNativeAPI.h"

#include <array>
#include <utility>

namespace nbl::video
{

struct CCUDAHandler::SNativeState
{
	struct SDeviceState
	{
		CUdevice handle = {};
		CUuuid uuid = {};
		std::array<int,CU_DEVICE_ATTRIBUTE_MAX> attributes = {};
	};

	cuda_native::CUDA cuda;
	cuda_native::NVRTC nvrtc;
	int cudaDriverVersion = 0;
	cuda_interop::SRuntimeVersion nvrtcVersion = {-1,-1};
	// Snapshot discovery at handler creation so diagnostics and NVRTC compile options describe the same runtime setup.
	cuda_interop::SRuntimeCompileEnvironment runtimeEnvironment;
	core::vector<std::string> runtimeIncludeOptions;
	core::vector<const char*> runtimeIncludeOptionPtrs;
	core::vector<SDeviceState> deviceStates;

	SNativeState(
		cuda_native::CUDA&& _cuda,
		cuda_native::NVRTC&& _nvrtc,
		int _cudaDriverVersion,
		cuda_interop::SRuntimeVersion _nvrtcVersion,
		cuda_interop::SRuntimeCompileEnvironment&& _runtimeEnvironment)
		: cuda(std::move(_cuda))
		, nvrtc(std::move(_nvrtc))
		, cudaDriverVersion(_cudaDriverVersion)
		, nvrtcVersion(_nvrtcVersion)
		, runtimeEnvironment(std::move(_runtimeEnvironment))
		, runtimeIncludeOptions(cuda_interop::makeNVRTCIncludeOptions(runtimeEnvironment))
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
