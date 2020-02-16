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
        uint32_t _ctxCount, uint32_t _ctxID, GLuint _GLname, COpenGLSpecializedShader::SProgramBinary&& _binary
    ) : IGPUComputePipeline(std::move(_layout), std::move(_cs)), 
        IOpenGLPipeline(_ctxCount, _ctxID, &_GLname, &_binary)
    {

    }

    bool containsShader() const { return static_cast<bool>(m_shader); }

    GLuint getShaderGLnameForCtx(uint32_t _stageIx, uint32_t _ctxID) const
    {
        if (GLuint n = IOpenGLPipeline<1>::getShaderGLnameForCtx(_stageIx, _ctxID))
            return n;

        const uint32_t name_ix = _ctxID;
        std::tie((*m_GLnames)[name_ix], (*m_shaderBinaries)[_stageIx]) =
            static_cast<const COpenGLSpecializedShader*>(m_shader.get())->compile(static_cast<const COpenGLPipelineLayout*>(getLayout()));

        return (*m_GLnames)[name_ix];
    }

protected:
    virtual ~COpenGLComputePipeline() = default;
};

}
}

#endif