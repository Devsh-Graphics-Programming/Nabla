#ifndef _NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED_


#include "nbl/video/IGPUShader.h"


namespace nbl::video
{

class ILogicalDevice;

class CVulkanShader : public IGPUShader
{
    public:
        CVulkanShader(const ILogicalDevice* dev, const E_SHADER_STAGE stage, std::string&& filepathHint, const VkShaderModule vk_shaderModule) :
            IGPUShader(core::smart_refctd_ptr<const ILogicalDevice>(dev), stage, std::move(filepathHint)), m_vkShaderModule(vk_shaderModule) {}

        inline VkShaderModule getInternalObject() const { return m_vkShaderModule; }

    private:
        ~CVulkanShader();

        VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;

};

}
#endif
