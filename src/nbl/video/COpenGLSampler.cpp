#include "nbl/video/COpenGLSampler.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLSampler::~COpenGLSampler()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroySampler(m_GLname);
}

}
