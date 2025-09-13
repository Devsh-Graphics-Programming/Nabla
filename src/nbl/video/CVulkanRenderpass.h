#ifndef _NBL_C_VULKAN_RENDERPASS_H_INCLUDED_
#define _NBL_C_VULKAN_RENDERPASS_H_INCLUDED_

#include "nbl/video/IGPURenderpass.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanRenderpass final : public IGPURenderpass
{
    public:
        inline explicit CVulkanRenderpass(const ILogicalDevice* logicalDevice, const SCreationParams& params, const SCreationParamValidationResult& counts, VkRenderPass vk_renderpass)
            : IGPURenderpass(core::smart_refctd_ptr<const ILogicalDevice>(logicalDevice),params,counts), m_renderpass(vk_renderpass) {}

        ~CVulkanRenderpass();

        VkRenderPass getInternalObject() const { return m_renderpass; }

        void setObjectDebugName(const char* label) const override;

    private:
        VkRenderPass m_renderpass;
};

}

#endif
