#include "nbl/video/COpenGLImageView.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLImageView::~COpenGLImageView()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyTexture(name);
}

}
}
