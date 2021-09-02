#ifndef __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__

#include "nbl/video/IGPUShader.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanShader : public IGPUShader
{
public:
    CVulkanShader(core::smart_refctd_ptr<ILogicalDevice>&& dev,
        core::smart_refctd_ptr<asset::ICPUBuffer>&& spirv)
        : IGPUShader(std::move(dev)), m_code(std::move(spirv)), m_containsGLSL(false)
    {}

    CVulkanShader(core::smart_refctd_ptr<ILogicalDevice>&& dev,
        core::smart_refctd_ptr<asset::ICPUBuffer>&& glslSource, buffer_contains_glsl_t buffer_contains_glsl)
        : IGPUShader(std::move(dev)), m_code(std::move(glslSource)), m_containsGLSL(true)
    {}

    const asset::ICPUBuffer* getSPVorGLSL() const { return m_code.get(); };
    const core::smart_refctd_ptr<asset::ICPUBuffer>& getSPVorGLSL_refctd() const { return m_code; };
    bool containsGLSL() const { return m_containsGLSL; }

private:
    //! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
    core::smart_refctd_ptr<asset::ICPUBuffer>	m_code;
    const bool									m_containsGLSL;
};
}

#define __NBL_VIDEO_C_VULKAN_SHADER_H_INCLUDED__
#endif
