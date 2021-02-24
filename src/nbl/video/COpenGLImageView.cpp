#include "nbl/video/COpenGLImageView.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLImageView::~COpenGLImageView()
{
    m_device->destroyTexture(name);
}

}
}

#endif