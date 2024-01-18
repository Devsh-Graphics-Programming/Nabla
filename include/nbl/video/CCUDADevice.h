// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_DEVICE_H_
#define _NBL_VIDEO_C_CUDA_DEVICE_H_


#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/CCUDASharedMemory.h"
#include "nbl/video/CCUDASharedSemaphore.h"

#ifdef _NBL_COMPILE_WITH_CUDA_

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
	#error "Need CUDA 9.0 SDK or higher."
#endif

// useful includes in the future
//#include "cudaEGL.h"
//#include "cudaVDPAU.h"

namespace nbl::video
{
class CCUDAHandler;
class CCUDASharedMemory;
class CCUDASharedSemaphore;

class CCUDADevice : public core::IReferenceCounted
{
    public:
#ifdef _WIN32
		static constexpr IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE EXTERNAL_MEMORY_HANDLE_TYPE = IDeviceMemoryAllocation::EHT_OPAQUE_WIN32;
		static constexpr CUmemAllocationHandleType ALLOCATION_HANDLE_TYPE = CU_MEM_HANDLE_TYPE_WIN32;
#else
		static constexpr IDeviceMemoryBacked::E_EXTERNAL_HANDLE_TYPE EXTERNAL_MEMORY_HANDLE_TYPE = IDeviceMemoryBacked::EHT_OPAQUE_FD;
		static constexpr CUmemAllocationHandleType ALLOCATION_TYPE = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
		struct SCUDACleaner : video::ICleanup
		{
			core::smart_refctd_ptr<const core::IReferenceCounted> resource;
			SCUDACleaner(core::smart_refctd_ptr<const core::IReferenceCounted> resource)
				: resource(std::move(resource))
			{ }
		};

		enum E_VIRTUAL_ARCHITECTURE
		{
			EVA_30,
			EVA_32,
			EVA_35,
			EVA_37,
			EVA_50,
			EVA_52,
			EVA_53,
			EVA_60,
			EVA_61,
			EVA_62,
			EVA_70,
			EVA_72,
			EVA_75,
			EVA_80,
			EVA_COUNT
		};
		static inline constexpr const char* virtualArchCompileOption[] = {
			"-arch=compute_30",
			"-arch=compute_32",
			"-arch=compute_35",
			"-arch=compute_37",
			"-arch=compute_50",
			"-arch=compute_52",
			"-arch=compute_53",
			"-arch=compute_60",
			"-arch=compute_61",
			"-arch=compute_62",
			"-arch=compute_70",
			"-arch=compute_72",
			"-arch=compute_75",
			"-arch=compute_80"
		};
		inline E_VIRTUAL_ARCHITECTURE getVirtualArchitecture() {return m_virtualArchitecture;}

		inline core::SRange<const char* const> geDefaultCompileOptions() const
		{
			return {m_defaultCompileOptions.data(),m_defaultCompileOptions.data()+m_defaultCompileOptions.size()};
		}

		// TODO/REDO Vulkan: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vulkan-interoperability
		// Watch out, use Driver API (`cu` functions) NOT the Runtime API (`cuda` functions)
		// Also maybe separate this out into its own `CCUDA` class instead of nesting it here?

		CUdevice getInternalObject() const { return m_handle; }
		const CCUDAHandler* getHandler() const { return m_handler.get();  }
		CUresult importGPUSemaphore(core::smart_refctd_ptr<CCUDASharedSemaphore>* outPtr, ISemaphore* sem);
		CUresult createSharedMemory(core::smart_refctd_ptr<CCUDASharedMemory>* outMem, struct CCUDASharedMemory::SCreationParams&& inParams);
		bool isMatchingDevice(const IPhysicalDevice* device) { return device && !memcmp(device->getProperties().deviceUUID, m_vulkanDevice->getProperties().deviceUUID, 16); }
		
		size_t roundToGranularity(CUmemLocationType location, size_t size) const;

	protected:
		CUresult reserveAdrressAndMapMemory(CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemLocationType location, CUmemGenericAllocationHandle memory);


		// CUDAHandler creates CUDADevice, it needs to access ctor
		friend class CCUDAHandler;

		CCUDADevice(core::smart_refctd_ptr<CVulkanConnection>&& _vulkanConnection, IPhysicalDevice* const _vulkanDevice, const E_VIRTUAL_ARCHITECTURE _virtualArchitecture, CUdevice _handle, core::smart_refctd_ptr<CCUDAHandler>&& _handler);
		~CCUDADevice();
		
		std::vector<const char*> m_defaultCompileOptions;
		core::smart_refctd_ptr<CVulkanConnection> m_vulkanConnection;
		IPhysicalDevice* const m_vulkanDevice;
		E_VIRTUAL_ARCHITECTURE m_virtualArchitecture;
		core::smart_refctd_ptr<CCUDAHandler> m_handler;
		CUdevice m_handle;
		CUcontext m_context;
		size_t m_allocationGranularity[4];
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif