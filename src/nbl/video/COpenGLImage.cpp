#include "nbl/video/COpenGLImage.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLImage::~COpenGLImage()
{
    m_device->destroyTexture(name);
}

}
}

#endif