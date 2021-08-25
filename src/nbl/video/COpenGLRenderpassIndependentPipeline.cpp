#include "nbl/video/COpenGLRenderpassIndependentPipeline.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLRenderpassIndependentPipeline::~COpenGLRenderpassIndependentPipeline()
{
    m_device->destroyPipeline(this);

    constexpr uint32_t MaxNamesPerStage = 128u;
    const auto namesPerStage = m_GLprograms->size() / SHADER_STAGE_COUNT;
    assert(namesPerStage < MaxNamesPerStage);

    GLuint names[MaxNamesPerStage]{};
    for (uint32_t i = 0u; i < SHADER_STAGE_COUNT; ++i)
    {
        if (!getShaderAtIndex(i))
            continue;

        for (uint32_t ctxid = 0u; ctxid < namesPerStage; ++ctxid)
            names[ctxid] = getShaderGLnameForCtx(i, ctxid);

        m_device->destroySpecializedShader(namesPerStage, names);
    }
}

}

#endif