#ifndef __NBL_VIDEO_C_VULKAN_SPECIALIZED_SHADER_H_INCLUDED__

#include "nbl/video/IGPUSpecializedShader.h"

#include <volk.h>

namespace nbl::video
{

class CVulkanSpecializedShader : public IGPUSpecializedShader
{
public:
    CVulkanSpecializedShader(
        core::smart_refctd_ptr<ILogicalDevice>&& dev,
        asset::IShader::E_SHADER_STAGE shaderStage, 
        core::smart_refctd_ptr<const CVulkanShader>&& unspecShader)
        : IGPUSpecializedShader(std::move(dev), shaderStage), m_unspecShader(std::move(unspecShader))
    {}

    inline VkShaderModule getInternalObject() const { return m_unspecShader->getInternalObject(); }

private:
    core::smart_refctd_ptr<const CVulkanShader> m_unspecShader;
};

}

#define __NBL_VIDEO_C_VULKAN_SPECIALIZED_SHADER_H_INCLUDED__
#endif

