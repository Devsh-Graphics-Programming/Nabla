#include "nbl/video/COpenGLImage.h"
#include "nbl/video/IOpenGL_LogicalDevice.h"
#include "nbl/video/COpenGLFramebuffer.h"

namespace nbl::video
{

COpenGLImage::~COpenGLImage()
{
    preDestroyStep();

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    // temporary fbos are created in the background to perform blits and color clears
    COpenGLFramebuffer::hash_t fbohash;
    if (asset::isDepthOrStencilFormat(m_creationParams.format))
        fbohash = COpenGLFramebuffer::getHashDepthStencilImage(this);
    else
        fbohash = COpenGLFramebuffer::getHashColorImage(this);
    device->destroyFramebuffer(fbohash);
    // destroy only if not observing (we own)
    if (!m_cachedCreationParams.skipHandleDestroy)
        device->destroyTexture(name);
}

void COpenGLImage::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_TEXTURE, name, strlen(getObjectDebugName()), getObjectDebugName());
}

}