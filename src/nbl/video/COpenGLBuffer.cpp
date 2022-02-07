#include "nbl/video/COpenGLBuffer.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{
COpenGLBuffer::~COpenGLBuffer()
{
    if(BufferName == 0u)
        return;
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyBuffer(BufferName);
}

void COpenGLBuffer::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->setObjectDebugName(GL_BUFFER, BufferName, strlen(getObjectDebugName()), getObjectDebugName());
}

}

#endif
