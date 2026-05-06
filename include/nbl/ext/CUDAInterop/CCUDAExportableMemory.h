// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_EXPORTABLE_MEMORY_H_
#define _NBL_VIDEO_C_CUDA_EXPORTABLE_MEMORY_H_

#include "nbl/video/declarations.h"

#include <memory>
#include <utility>

namespace nbl::video
{
class CCUDADevice;

namespace cuda_native
{
struct SAccess;
}

enum class ECUDAMemoryLocation : uint32_t
{
	DEVICE = 1,
	HOST = 2,
	HOST_NUMA = 3,
	HOST_NUMA_CURRENT = 4
};

class CCUDAExportableMemory : public core::IReferenceCounted
{
	public:
		struct SNativeState;
		struct SCreationParams
		{
			size_t size;
			uint32_t alignment;
			ECUDAMemoryLocation location;
		};

		struct SCachedCreationParams : SCreationParams
		{
			size_t granularSize;
			external_handle_t externalHandle;
		};

		CCUDAExportableMemory(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params, std::unique_ptr<SNativeState>&& nativeState);
		~CCUDAExportableMemory() override;

		const SCreationParams& getCreationParams() const { return m_params; }

		core::smart_refctd_ptr<IDeviceMemoryAllocation> exportAsMemory(ILogicalDevice* device, IDeviceMemoryBacked* dedication = nullptr) const;

	private:
		friend struct cuda_native::SAccess;

		core::smart_refctd_ptr<CCUDADevice> m_device;
		SCachedCreationParams m_params;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
