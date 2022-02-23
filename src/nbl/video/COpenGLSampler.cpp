#include "nbl/video/COpenGLSampler.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

COpenGLSampler::~COpenGLSampler()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroySampler(m_GLname);
}

void COpenGLSampler::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_SAMPLER, m_GLname, strlen(getObjectDebugName()), getObjectDebugName());
}

}
