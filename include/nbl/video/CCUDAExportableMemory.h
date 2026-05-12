// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_EXPORTABLE_MEMORY_H_
#define _NBL_VIDEO_C_CUDA_EXPORTABLE_MEMORY_H_

#include "nbl/video/declarations.h"
#include "nbl/video/CUDAInteropHandles.h"

#include <memory>
#include <utility>

namespace nbl::video
{
class CCUDADevice;

class NBL_API2 CCUDAExportableMemory final : public core::IReferenceCounted
{
	public:
		struct SCachedCreationParams
		{
			size_t granularSize;
			system::external_handle_t externalHandle;
			bool deviceLocal;
		};

		~CCUDAExportableMemory() override;

		cuda_interop::SCUdeviceptr getDeviceptr() const;

		/**
		 * @brief Exports the CUDA memory as a Vulkan device memory allocation.
		 * 
		 * Creates an IDeviceMemoryAllocation object that references the underlying CUDA memory,
		 * allowing it to be used within the Vulkan rendering pipeline while maintaining
		 * interoperability with CUDA operations.
		 * 
		 * @param device The logical device that will own the exported memory allocation.
		 * @param dedication Optional pointer to a device memory backed resource for dedicated allocation.
		 *                   If provided, the memory will be dedicated to that specific resource and
		 *                   automatically bound to it.
		 * @return A smart pointer to the exported IDeviceMemoryAllocation, or nullptr on failure.
		 */
		core::smart_refctd_ptr<IDeviceMemoryAllocation> exportAsMemory(ILogicalDevice* device, IDeviceMemoryBacked* dedication = nullptr) const;

	private:
		friend class CCUDADevice;

		struct SNativeState;
		CCUDAExportableMemory(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params, std::unique_ptr<SNativeState>&& nativeState);
		static core::smart_refctd_ptr<CCUDAExportableMemory> create(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params, std::unique_ptr<SNativeState>&& nativeState);

		core::smart_refctd_ptr<CCUDADevice> m_device;
		SCachedCreationParams m_params;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
