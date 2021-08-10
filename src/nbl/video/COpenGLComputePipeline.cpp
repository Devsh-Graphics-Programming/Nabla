#include "nbl/video/COpenGLComputePipeline.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLComputePipeline::~COpenGLComputePipeline()
{
    constexpr uint32_t MaxNames = 128u;
    GLuint names[MaxNames]{};

    const auto namesCount = m_GLprograms->size();
    assert(namesCount < MaxNames);
    for (uint32_t i = 0u; i < namesCount; ++i)
        names[i] = getShaderGLnameForCtx(0u, i);

    m_device->destroySpecializedShader(namesCount, names);
}

}

#endif