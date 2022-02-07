#include "nbl/video/COpenGLFramebuffer.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{
COpenGLFramebuffer::COpenGLFramebuffer(core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev, SCreationParams&& params)
    : IGPUFramebuffer(core::smart_refctd_ptr(dev), std::move(params)), m_device(dev.get())
{
}

COpenGLFramebuffer::~COpenGLFramebuffer()
{
    m_device->destroyFramebuffer(getHashValue());
}

}