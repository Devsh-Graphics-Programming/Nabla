#ifndef _NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED_

#include "nbl/video/IGPUGraphicsPipeline.h"

namespace nbl::video
{

class CVulkanGraphicsPipeline final : public IGPUGraphicsPipeline
{
    public:
        CVulkanGraphicsPipeline(const ILogicalDevice* dev, SCreationParams&& params, const VkPipeline vk_pipeline) :
            IGPUGraphicsPipeline(core::smart_refctd_ptr<const ILogicalDevice>(dev),std::move(params)), m_vkPipeline(vk_pipeline) {}

        inline VkPipeline getInternalObject() const { return m_vkPipeline; }

    private:
        ~CVulkanGraphicsPipeline();

        const VkPipeline m_vkPipeline;
        // gotta keep those VkShaderModules alive (for now)
        core::smart_refctd_ptr<const CVulkanShader> m_shaders[IGPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT];
};

}

#endif
