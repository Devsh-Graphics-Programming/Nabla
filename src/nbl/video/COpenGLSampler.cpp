#include "nbl/video/COpenGLSampler.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLSampler::~COpenGLSampler()
{
    m_device->destroySampler(m_GLname);
}

}
}

#endif