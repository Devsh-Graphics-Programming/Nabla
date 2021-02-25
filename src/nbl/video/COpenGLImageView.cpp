#include "nbl/video/COpenGLImageView.h"

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
