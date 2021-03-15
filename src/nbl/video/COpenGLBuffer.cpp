#include "COpenGLBuffer.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

std::atomic_uint32_t COpenGLBuffer::s_reallocCounter = 0u;

COpenGLBuffer::~COpenGLBuffer()
{
    auto* device = static_cast<IOpenGL_LogicalDevice*>(const_cast<ILogicalDevice*>(getOriginDevice()));
    device->destroyBuffer(BufferName);
}

}
}

#endif
