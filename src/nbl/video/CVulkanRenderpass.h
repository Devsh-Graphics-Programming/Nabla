#ifndef __NBL_C_VULKAN_RENDERPASS_H_INCLUDED__
#define __NBL_C_VULKAN_RENDERPASS_H_INCLUDED__

#include "nbl/video/IGPURenderpass.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanRenderpass final : public IGPURenderpass
{
public:
    explicit CVulkanRenderpass(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
        const SCreationParams& params, VkRenderPass vk_renderpass)
        : IGPURenderpass(std::move(logicalDevice), params), m_renderpass(vk_renderpass)
    {}

    ~CVulkanRenderpass();

    VkRenderPass getInternalObject() const { return m_renderpass; }

    void setObjectDebugName(const char* label) const override;

private:
    VkRenderPass m_renderpass;
};

}

#endif
