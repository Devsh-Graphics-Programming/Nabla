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
		CVulkanImage(core::smart_refctd_ptr<ILogicalDevice>&& _vkdev,
			IGPUImage::SCreationParams&& _params, VkImage _vkimg,
			const IDriverMemoryBacked::SDriverMemoryRequirements& reqs = IDriverMemoryBacked::SDriverMemoryRequirements())
			: IGPUImage(std::move(_vkdev), std::move(_params), reqs), m_vkImage(_vkimg)
		{}

		inline VkImage getInternalObject() const { return m_vkImage; }

		inline IDriverMemoryAllocation* getBoundMemory() override { return m_memory.get(); }

		inline const IDriverMemoryAllocation* getBoundMemory() const override { return m_memory.get(); }

		inline size_t getBoundMemoryOffset() const override { return m_memBindingOffset; }

		inline void setMemoryAndOffset(core::smart_refctd_ptr<IDriverMemoryAllocation>&& memory, uint64_t memBindingOffset)
		{
			m_memory = std::move(memory);
			m_memBindingOffset = memBindingOffset;
		}

	protected:
		virtual ~CVulkanImage();

		core::smart_refctd_ptr<IDriverMemoryAllocation> m_memory = nullptr;
		uint64_t m_memBindingOffset;
		VkImage m_vkImage;
};

} // end namespace nbl::video

#endif
