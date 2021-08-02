// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_VULKAN_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_C_VULKAN_IMAGE_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IGPUImage.h"
#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

class CVulkanImage final : public IGPUImage
{
	protected:
		virtual ~CVulkanImage();

	private:
		CVKLogicalDevice* m_vkdevice;
		VkImage m_vkimg;
		core::smart_refctd_ptr<CVKSwapchain> m_swapchain;

	public:
		//! constructor
		CVulkanImage(CVKLogicalDevice* _vkdev, IGPUImage::SCreationParams&& _params);
		CVulkanImage(CVKLogicalDevice* _vkdev, IGPUImage::SCreationParams&& _params, VkImage _vkimg) :
			IGPUImage(_vkdev, std::move(_params)), m_vkdevice(_vkdev), m_vkimg(_vkimg)
		{

		}

		inline VkImage getInternalObject() const { return m_vkimg; }

		// Todo(achal): Gotta move move this into a new file CVulkanMemoryAllocation
		// inline size_t getAllocationSize() const override { return this->getImageDataSizeInBytes(); }
		inline IDriverMemoryAllocation* getBoundMemory() override { return nullptr; }
		inline const IDriverMemoryAllocation* getBoundMemory() const override { return nullptr; }
		inline size_t getBoundMemoryOffset() const override { return 0ull; }

		// inline E_SOURCE_MEMORY_TYPE getType() const override { return ESMT_DEVICE_LOCAL; }
		// This exists as a pure virtual function in ILogicalDevice which takes in a IDriverMemoryAllocation* --which is not a base class of this class
		// inline void unmapMemory() override {}
		// inline bool isDedicated() const override { return true; }
};

} // end namespace nbl::video

#endif
