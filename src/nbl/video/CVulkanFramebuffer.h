#ifndef __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__

#include "nbl/video/IGPUFramebuffer.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanFramebuffer final : public IGPUFramebuffer
{
public:
    CVulkanFramebuffer(core::smart_refctd_ptr<ILogicalDevice>&& dev, SCreationParams&& params,
        VkFramebuffer vk_framebuffer)
        : IGPUFramebuffer(std::move(dev), std::move(params)), m_vkfbo(vk_framebuffer)
    {}

    ~CVulkanFramebuffer();

    inline VkFramebuffer getInternalObject() const { return m_vkfbo; }

private:
    VkFramebuffer m_vkfbo;
};

}

#endif