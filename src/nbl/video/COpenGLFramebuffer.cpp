#include "nbl/video/COpenGLFramebuffer.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLFramebuffer::COpenGLFramebuffer(IOpenGL_LogicalDevice* dev, SCreationParams&& params) : IGPUFramebuffer(dev, std::move(params)), m_device(dev)
{
}

COpenGLFramebuffer::~COpenGLFramebuffer()
{
    m_device->destroyFramebuffer(getHashValue());
}

}