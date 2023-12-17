#ifndef _NBL_VIDEO_C_VULKAN_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_

#include "nbl/video/IGPURenderpassIndependentPipeline.h"

namespace nbl::video
{

class CVulkanRenderpassIndependentPipeline : public IGPURenderpassIndependentPipeline
{
	public:
		CVulkanRenderpassIndependentPipeline(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SCreationParams& params) :
			IGPURenderpassIndependentPipeline(std::move(dev),params)
		{
			for (const auto& info : params.shaders)
			if (info.shader)
			{
				const auto stageIx = core::findLSB(info.shader->getStage());
				m_shaders[stageIx] = core::smart_refctd_ptr<const IGPUShader>(info.shader);
			}
		}

	private:
		// gotta keep those VkShaderModules alive (for now)
		core::smart_refctd_ptr<const IGPUShader> m_shaders[GRAPHICS_SHADER_STAGE_COUNT];
};

}

#endif
