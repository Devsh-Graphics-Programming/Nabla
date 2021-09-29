#include "nbl/video/COpenGLImageView.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLImageView::~COpenGLImageView()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyTexture(name);
}

void COpenGLImageView::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_TEXTURE, name, strlen(getObjectDebugName()), getObjectDebugName());
}

}
