#include "COpenGLBuffer.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

COpenGLBuffer::~COpenGLBuffer()
{
    m_device->destroyBuffer(BufferName);
}

}
}

#endif
