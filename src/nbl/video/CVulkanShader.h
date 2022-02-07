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
        core::smart_refctd_ptr<asset::ICPUBuffer>&& spirv,
        const E_SHADER_STAGE stage,
        std::string&& filepathHint,
        VkShaderModule vk_shaderModule)
        : IGPUShader(std::move(dev), stage, std::move(filepathHint)),
          m_spirv(std::move(spirv)), m_vkShaderModule(vk_shaderModule)
    {}

    ~CVulkanShader();

    const asset::ICPUBuffer* getSPV() const { return m_spirv.get(); };

    inline VkShaderModule getInternalObject() const { return m_vkShaderModule; }

private:
    core::smart_refctd_ptr<asset::ICPUBuffer> m_spirv;
    VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
};
}

#define __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__
#endif
