#include "nbl/video/COpenGLRenderpassIndependentPipeline.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{
COpenGLRenderpassIndependentPipeline::~COpenGLRenderpassIndependentPipeline()
{
    m_device->destroyPipeline(this);

    m_device->destroySpecializedShaders(std::move(m_GLprograms));
}

}

#endif