#ifndef __IRR_C_OPENGL_COMPUTE_PIPELINE_H_INCLUDED__
#define __IRR_C_OPENGL_COMPUTE_PIPELINE_H_INCLUDED__

#include "irr/video/IGPUComputePipeline.h"
#include "irr/video/COpenGLSpecializedShader.h"

namespace irr { 
namespace video
{

class COpenGLComputePipeline : public IGPUComputePipeline
{
public:
    using IGPUComputePipeline::IGPUComputePipeline;

    //! @returns separable shader program name
    inline GLuint getGLnameForCtx(uint32_t _ctxID) const
    {
        return static_cast<const COpenGLSpecializedShader*>(m_shader.get())->getGLnameForCtx(_ctxID);
    }

protected:
    virtual ~COpenGLComputePipeline() = default;
};

}
}

#endif