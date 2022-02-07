#include "nbl/video/COpenGLComputePipeline.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{
COpenGLComputePipeline::~COpenGLComputePipeline()
{
    m_device->destroySpecializedShaders(std::move(m_GLprograms));
}

}

#endif