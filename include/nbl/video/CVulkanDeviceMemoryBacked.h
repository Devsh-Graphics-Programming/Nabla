// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_VULKAN_DEVICE_MEMORY_BACKED_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_DEVICE_MEMORY_BACKED_H_INCLUDED_

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/IGPUImage.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class CVulkanLogicalDevice;

template<class Interface>
class CVulkanDeviceMemoryBacked : public Interface
{
		constexpr static inline bool IsImage = std::is_same_v<Interface,IGPUImage>;
		using VkResource_t = const std::conditional_t<IsImage,VkImage,VkBuffer>;

	public:
		inline IDeviceMemoryBacked::SMemoryBinding getBoundMemory() const {return {m_memory.get(),m_offset};}
		inline void setMemoryBinding(const IDeviceMemoryBacked::SMemoryBinding& binding)
		{
			m_memory = core::smart_refctd_ptr<IDeviceMemoryAllocation>(binding.memory);
			m_offset = binding.offset;
		}

		inline const void* getNativeHandle() const override {return &m_handle;}
		inline VkResource_t getInternalObject() const {return m_handle;}

	protected:
		// special constructor for when memory requirements are known up-front (so far only swapchains and internal forwarding here)
		CVulkanDeviceMemoryBacked(const CVulkanLogicalDevice* dev, Interface::SCreationParams&& _creationParams, const IDeviceMemoryBacked::SDeviceMemoryRequirements& _memReqs, const VkResource_t vkHandle);
		CVulkanDeviceMemoryBacked(const CVulkanLogicalDevice* dev, Interface::SCreationParams&& _creationParams, const VkResource_t vkHandle) :
			CVulkanDeviceMemoryBacked(dev,std::move(_creationParams),obtainRequirements(dev,vkHandle),vkHandle) {}

	private:
		static IDeviceMemoryBacked::SDeviceMemoryRequirements obtainRequirements(const CVulkanLogicalDevice* device, const VkResource_t vkHandle);

		core::smart_refctd_ptr<IDeviceMemoryAllocation> m_memory = nullptr;
		size_t m_offset = 0u;
		const VkResource_t m_handle;
};

#ifndef _NBL_VIDEO_C_VULKAN_DEVICE_MEMORY_BACKED_CPP_
extern template CVulkanDeviceMemoryBacked<IGPUBuffer>;
extern template CVulkanDeviceMemoryBacked<IGPUImage>;
#endif

} // end namespace nbl::video

#endif
