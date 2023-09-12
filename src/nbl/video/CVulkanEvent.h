#ifndef _NBL_VIDEO_C_VULKAN_EVENT_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_EVENT_H_INCLUDED_

#include "nbl/video/IEvent.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanEvent final : public IEvent
{
    public:
        inline CVulkanEvent(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const CREATE_FLAGS flags, const VkEvent vk_event)
            : IEvent(std::move(dev), flags), m_vkEvent(vk_event) {}
        ~CVulkanEvent();

        STATUS getEventStatus_impl() const override;
        STATUS resetEvent_impl() override;
        STATUS setEvent_impl() override;

        inline VkEvent getInternalObject() const { return m_vkEvent; }

    private:
        VkEvent m_vkEvent;
};

}

#endif