// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_VULKAN_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_C_VULKAN_IMAGE_H_INCLUDED__

#include "nbl/video/IGPUImage.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanImage : public IGPUImage
{
	public:
		// CVulkanImage(ILogicalDevice* _vkdev, IGPUImage::SCreationParams&& _params);
		CVulkanImage(core::smart_refctd_ptr<ILogicalDevice>&& _vkdev, IGPUImage::SCreationParams&& _params, VkImage _vkimg) :
			IGPUImage(std::move(_vkdev), std::move(_params)), m_vkImage(_vkimg)
		{}

		inline VkImage getInternalObject() const { return m_vkImage; }

		// Todo(achal)
		inline IDriverMemoryAllocation* getBoundMemory() override { return nullptr; }
		inline const IDriverMemoryAllocation* getBoundMemory() const override { return nullptr; }
		inline size_t getBoundMemoryOffset() const override { return 0ull; }

	protected:
		virtual ~CVulkanImage();

		VkImage m_vkImage;
};

} // end namespace nbl::video

#endif
