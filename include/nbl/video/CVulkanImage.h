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
		CVulkanImage(core::smart_refctd_ptr<const ILogicalDevice>&& _vkdev,
			IGPUImage::SCreationParams&& _params, VkImage _vkimg,
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs)
			: IGPUImage(std::move(_vkdev), reqs, std::move(_params)), m_vkImage(_vkimg)
		{}

		// foreign image
		// TODO investigation into VK/CUDA interop to refine this
		CVulkanImage(core::smart_refctd_ptr<const ILogicalDevice>&& _vkdev,
			IGPUImage::SCreationParams&& _params, VkImage _vkimg,
			core::smart_refctd_ptr<ISwapchain> _backingSwapchain = nullptr,
			uint32_t _backingSwapchainIx = 0)
			: IGPUImage(std::move(_vkdev), std::move(_params), _backingSwapchain, _backingSwapchainIx), m_vkImage(_vkimg)
		{}

		inline const void* getNativeHandle() const override {return &m_vkImage;}
		inline VkImage getInternalObject() const { return m_vkImage; }

		inline IDeviceMemoryAllocation* getBoundMemory() override { return m_memory.get(); }

		inline const IDeviceMemoryAllocation* getBoundMemory() const override { return m_memory.get(); }

		inline size_t getBoundMemoryOffset() const override { return m_memBindingOffset; }

		inline void setMemoryAndOffset(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& memory, uint64_t memBindingOffset)
		{
			m_memory = std::move(memory);
			m_memBindingOffset = memBindingOffset;
		}

		void setObjectDebugName(const char* label) const override;

	protected:
		virtual ~CVulkanImage();

		core::smart_refctd_ptr<IDeviceMemoryAllocation> m_memory = nullptr;
		uint64_t m_memBindingOffset;
		VkImage m_vkImage;
};

} // end namespace nbl::video

#endif
