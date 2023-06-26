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

class ILogicalDevice;

template<class Interface>
class CVulkanDeviceMemoryBacked : public Interface
{
		constexpr static inline bool IsImage = std::is_same_v<Interface,IGPUImage>;
		using VkResource_t = std::conditional_t<IsImage,VkImage,VkBuffer>;
		static IDeviceMemoryBacked::SDeviceMemoryRequirements obtainRequirements(const ILogicalDevice* device, const VkResource_t vkHandle);

		core::smart_refctd_ptr<IDeviceMemoryAllocation> m_memory;
		VkResource_t m_handle;

	public:
		inline IDeviceMemoryAllocation* getBoundMemory() override {return m_memory.get();}

		inline void* getNativeHandle() override {return &m_handle;}
		inline VkResource_t getInternalObject() const {return m_handle;}

	protected:
		inline CVulkanDeviceMemoryBacked(const ILogicalDevice* dev, Interface::SCreationParams&& _creationParams, const VkResource_t vkHandle)
			: Interface(core::smart_refctd_ptr<const ILogicalDevice>(dev),std::move(creationParams),getRequirements(dev)), m_handle(vkHandle)
		{
			assert(vkHandle!=VK_NULL_HANDLE);
		}
};

#ifndef _NBL_VIDEO_C_VULKAN_DEVICE_MEMORY_BACKED_CPP_
extern template CVulkanDeviceMemoryBacked<IGPUBuffer>;
extern template CVulkanDeviceMemoryBacked<IGPUImage>;
#endif

} // end namespace nbl::video

#endif
