#include "nbl/video/COpenGLBufferView.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLBufferView::~COpenGLBufferView()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyTexture(m_textureName);
}

}
}

#endif