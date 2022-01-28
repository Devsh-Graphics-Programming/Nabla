#ifndef __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__

#include "nbl/video/IGPUShader.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanShader : public IGPUShader
{
public:
    CVulkanShader(
        core::smart_refctd_ptr<ILogicalDevice>&& dev,
        const E_SHADER_STAGE stage,
        std::string&& filepathHint,
        VkShaderModule vk_shaderModule)
        : IGPUShader(std::move(dev), stage, std::move(filepathHint))
        , m_vkShaderModule(vk_shaderModule)
    {
    }

    ~CVulkanShader();

    inline VkShaderModule getInternalObject() const { return m_vkShaderModule; }

private:
    VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;

};
}

#define __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__
#endif
