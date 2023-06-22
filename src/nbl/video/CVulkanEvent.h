#ifndef _NBL_C_VULKAN_EVENT_H_INCLUDED_
#define _NBL_C_VULKAN_EVENT_H_INCLUDED_

#include "nbl/video/IGPUEvent.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanEvent final : public IGPUEvent
{
    public:
        inline CVulkanEvent(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const E_CREATE_FLAGS flags, const VkEvent vk_event)
            : IGPUEvent(std::move(dev), flags), m_vkEvent(vk_event) {}

        ~CVulkanEvent();

        inline VkEvent getInternalObject() const { return m_vkEvent; }

    private:
        VkEvent m_vkEvent;
};

}

#endif