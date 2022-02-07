#ifndef __NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED__
#define __NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED__

#include "nbl/video/IGPUGraphicsPipeline.h"

namespace nbl::video
{
class CVulkanGraphicsPipeline : public IGPUGraphicsPipeline
{
public:
    CVulkanGraphicsPipeline(
        core::smart_refctd_ptr<const ILogicalDevice>&& dev,
        SCreationParams&& params,
        VkPipeline vk_pipline)
        : IGPUGraphicsPipeline(std::move(dev), std::move(params)), m_vkPipeline(vk_pipline)
    {}

    ~CVulkanGraphicsPipeline();

    inline VkPipeline getInternalObject() const { return m_vkPipeline; }

private:
    VkPipeline m_vkPipeline;
};

}

#endif
