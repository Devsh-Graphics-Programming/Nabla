// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_DEVICE_H_
#define _NBL_VIDEO_C_CUDA_DEVICE_H_

#include "nbl/video/declarations.h"
#include "nbl/video/CUDAInteropHandles.h"
#include "nbl/video/CCUDAExportableMemory.h"
#include "nbl/video/CCUDAImportedMemory.h"
#include "nbl/video/CCUDAImportedSemaphore.h"

#include <array>
#include <memory>
#include <vector>

namespace nbl::video
{
class CCUDAHandler;

class NBL_API2 CCUDADevice : public core::IReferenceCounted
{
	public:
#ifdef _WIN32
		static constexpr IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE EXTERNAL_MEMORY_HANDLE_TYPE = IDeviceMemoryAllocation::EHT_OPAQUE_WIN32;
#else
		static constexpr IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE EXTERNAL_MEMORY_HANDLE_TYPE = IDeviceMemoryAllocation::EHT_OPAQUE_FD;
#endif

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
			EVA_86,
			EVA_87,
			EVA_88,
			EVA_89,
			EVA_90,
			EVA_90A,
			EVA_100,
			EVA_100A,
			EVA_100F,
			EVA_103,
			EVA_103A,
			EVA_103F,
			EVA_110,
			EVA_110A,
			EVA_110F,
			EVA_120,
			EVA_120A,
			EVA_120F,
			EVA_121,
			EVA_121A,
			EVA_121F,
			EVA_COUNT
		};
		E_VIRTUAL_ARCHITECTURE getVirtualArchitecture() const;


		core::SRange<const char* const> geDefaultCompileOptions() const;

		const CCUDAHandler* getHandler() const;
		cuda_interop::SCUdevice getInternalObject() const;
		cuda_interop::SCUcontext getContext() const;

		struct SExportableMemoryCreationParams
		{
			size_t size;
			uint32_t alignment;
			uint32_t locationType;
		};

		inline size_t roundToGranularity(uint32_t locationType, size_t size) const
		{
			if (locationType>=m_allocationGranularity.size())
				return 0u;
			const auto granularity = m_allocationGranularity[locationType];
			if (size==0u || granularity==0u)
				return 0u;
			return ((size - 1) / granularity + 1) * granularity;
		}

		core::smart_refctd_ptr<CCUDAExportableMemory> createExportableMemory(SExportableMemoryCreationParams&& params);
		core::smart_refctd_ptr<CCUDAImportedMemory> importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& mem);

		core::smart_refctd_ptr<CCUDAImportedSemaphore> importExternalSemaphore(core::smart_refctd_ptr<ISemaphore>&& sem);

	private:
		friend class CCUDAHandler;

		static constexpr uint32_t AllocationGranularityLocationTypeCount = 5u;
		struct SNativeState;
		CCUDADevice(core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, IPhysicalDevice* const vulkanDevice, const E_VIRTUAL_ARCHITECTURE virtualArchitecture, std::unique_ptr<SNativeState>&& nativeState, core::smart_refctd_ptr<CCUDAHandler>&& handler);
		~CCUDADevice() override;
		bool isValid() const;

		const system::logger_opt_ptr m_logger;
		std::vector<const char*> m_defaultCompileOptions;
		core::smart_refctd_ptr<CVulkanConnection> m_vulkanConnection;
		IPhysicalDevice* const m_physicalDevice;
		std::array<size_t,AllocationGranularityLocationTypeCount> m_allocationGranularity = {};
		E_VIRTUAL_ARCHITECTURE m_virtualArchitecture;
		bool m_valid = false;

		core::smart_refctd_ptr<CCUDAHandler> m_handler;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
