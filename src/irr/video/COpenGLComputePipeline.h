#ifndef __IRR_C_OPENGL_COMPUTE_PIPELINE_H_INCLUDED__
#define __IRR_C_OPENGL_COMPUTE_PIPELINE_H_INCLUDED__

#include "irr/video/IGPUComputePipeline.h"
#include "irr/video/IOpenGLPipeline.h"

namespace irr { 
namespace video
{

class COpenGLComputePipeline : public IGPUComputePipeline, public IOpenGLPipeline<1>
{
public:
    COpenGLComputePipeline(
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        core::smart_refctd_ptr<IGPUSpecializedShader>&& _cs,
        uint32_t _ctxCount, uint32_t _ctxID, GLuint _GLname, const COpenGLSpecializedShader::SProgramBinary& _binary
    ) : IGPUComputePipeline(std::move(_layout), std::move(_cs)), 
        IOpenGLPipeline(_ctxCount, _ctxID, &_GLname, &_binary)
    {

    }

    bool containsShader() const { return static_cast<bool>(m_shader); }

    GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
    {
        return IOpenGLPipeline<1>::getShaderGLnameForCtx(_stageIx, _ctxID);
    }

    void setUniformsImitatingPushConstants(uint32_t _ctxID, const uint8_t* _pcData) const
    {
        auto uniforms = static_cast<COpenGLSpecializedShader*>(m_shader.get())->getUniforms();
        auto locations = static_cast<COpenGLSpecializedShader*>(m_shader.get())->getLocations();
        if (!uniforms.length())
            return;

        IOpenGLPipeline<1>::setUniformsImitatingPushConstants(0u, _ctxID, _pcData, uniforms, locations);
    }

protected:
    virtual ~COpenGLComputePipeline() = default;
};

}
}

#endif