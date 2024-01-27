#ifndef _NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPUGraphicsPipeline.h"

#include "nbl/video/CVulkanShader.h"


namespace nbl::video
{

class CVulkanGraphicsPipeline final : public IGPUGraphicsPipeline
{
    public:
        CVulkanGraphicsPipeline(const SCreationParams& params, const VkPipeline vk_pipeline) :
            IGPUGraphicsPipeline(params), m_vkPipeline(vk_pipeline)
        {
			for (const auto& info : params.shaders)
			if (info.shader)
			{
				const auto stageIx = hlsl::findLSB(info.shader->getStage());
				m_shaders[stageIx] = core::smart_refctd_ptr<const CVulkanShader>(static_cast<const CVulkanShader*>(info.shader));
			}
        }

        inline const void* getNativeHandle() const override {return &m_vkPipeline;}

        inline VkPipeline getInternalObject() const {return m_vkPipeline;}

    private:
        ~CVulkanGraphicsPipeline();

        const VkPipeline m_vkPipeline;
        // gotta keep those VkShaderModules alive (for now)
        core::smart_refctd_ptr<const CVulkanShader> m_shaders[GRAPHICS_SHADER_STAGE_COUNT];
};

}

#endif
