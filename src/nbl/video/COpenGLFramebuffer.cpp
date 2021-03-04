#include "nbl/video/COpenGLFramebuffer.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLFramebuffer::COpenGLFramebuffer(SCreationParams&& params, IOpenGL_LogicalDevice* dev) : IGPUFramebuffer(dev, std::move(params)), m_device(dev)
{
}

COpenGLFramebuffer::~COpenGLFramebuffer()
{
    m_device->destroyFramebuffer(this);
}

}
}