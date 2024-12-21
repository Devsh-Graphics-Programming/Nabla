#ifndef _NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED_
#define _NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED_

#include "nbl/video/IGPUFramebuffer.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanFramebuffer final : public IGPUFramebuffer
{
    public:
        CVulkanFramebuffer(core::smart_refctd_ptr<ILogicalDevice>&& dev, SCreationParams&& params, const VkFramebuffer vk_framebuffer)
            : IGPUFramebuffer(std::move(dev),std::move(params)), m_vkfbo(vk_framebuffer) {}

        ~CVulkanFramebuffer();

        inline VkFramebuffer getInternalObject() const {return m_vkfbo;}

        void setObjectDebugName(const char* label) const override;

    private:
        const VkFramebuffer m_vkfbo;
};

}

#endif