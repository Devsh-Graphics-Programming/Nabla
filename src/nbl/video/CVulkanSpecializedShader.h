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
        core::smart_refctd_ptr<const CVulkanShader>&& unspecShader,
        const asset::ISpecializedShader::SInfo& specInfo)
        : IGPUSpecializedShader(std::move(dev), unspecShader->getStage()),
        m_unspecShader(std::move(unspecShader)), m_specInfo(specInfo)
    {}

    inline VkShaderModule getInternalObject() const { return m_unspecShader->getInternalObject(); }

    inline const asset::ISpecializedShader::SInfo& getSpecInfo() const { return m_specInfo; }

private:
    core::smart_refctd_ptr<const CVulkanShader> m_unspecShader;
    asset::ISpecializedShader::SInfo m_specInfo;
};

}

#define __NBL_VIDEO_C_VULKAN_SPECIALIZED_SHADER_H_INCLUDED__
#endif

