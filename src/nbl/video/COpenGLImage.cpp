#include "nbl/video/COpenGLImage.h"
#include "nbl/video/IOpenGL_LogicalDevice.h"
#include "nbl/video/COpenGLFramebuffer.h"

namespace nbl::video
{

COpenGLImage::~COpenGLImage()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyTexture(name);
    // temporary fbos are created in the background to perform blits and color clears
    COpenGLFramebuffer::hash_t fbohash;
    if (asset::isDepthOrStencilFormat(params.format))
        fbohash = COpenGLFramebuffer::getHashDepthStencilImage(this);
    else
        fbohash = COpenGLFramebuffer::getHashColorImage(this);
    device->destroyFramebuffer(fbohash);
}

void COpenGLImage::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_TEXTURE, name, getObjectDebugName().size(), getObjectDebugName().c_str());
}

}