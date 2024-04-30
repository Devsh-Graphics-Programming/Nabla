#ifndef _NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED_
#define _NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED_

#include "nbl/video/IGPUFramebuffer.h"

NBL_PUSH_DISABLE_WARNINGS
#include <volk.h>
NBL_POP_DISABLE_WARNINGS

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