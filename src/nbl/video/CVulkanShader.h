#ifndef __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__

#include "nbl/video/IGPUSpecializedshader.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanShader : public IGPUSpecializedShader
{
public:
    CVulkanShader(core::smart_refctd_ptr<ILogicalDevice>&& dev,
        asset::ISpecializedShader::E_SHADER_STAGE shaderStage, VkShaderModule vk_shaderModule)
        : IGPUSpecializedShader(std::move(dev), shaderStage), m_vkShaderModule(vk_shaderModule)
    {}

    ~CVulkanShader();

    inline VkShaderModule getInternalObject() const { return m_vkShaderModule; }

private:
    VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
};
}

#define __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__
#endif
