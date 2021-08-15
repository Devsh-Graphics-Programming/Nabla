#ifndef __NBL_VIDEO_C_VULKAN_SPECIALIZED_SHADER_H_INCLUDED__

#include "nbl/video/IGPUSpecializedShader.h"

#include <volk.h>

namespace nbl::video
{

class CVulkanSpecializedShader : public IGPUSpecializedShader
{
public:
    CVulkanSpecializedShader(core::smart_refctd_ptr<ILogicalDevice>&& dev, VkShaderModule shaderModule,
        asset::ISpecializedShader::E_SHADER_STAGE shaderStage)
        : IGPUSpecializedShader(std::move(dev), shaderStage), m_shaderModule(shaderModule)
    {}

    ~CVulkanSpecializedShader();

    inline VkShaderModule getInternalObject() const { return m_shaderModule; }

private:
    VkShaderModule m_shaderModule;
};

}

#define __NBL_VIDEO_C_VULKAN_SPECIALIZED_SHADER_H_INCLUDED__
#endif

