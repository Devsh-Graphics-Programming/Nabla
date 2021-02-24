#include "nbl/video/COpenGLBufferView.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLBufferView::~COpenGLBufferView()
{
    m_device->destroyTexture(m_textureName);
}

}
}

#endif