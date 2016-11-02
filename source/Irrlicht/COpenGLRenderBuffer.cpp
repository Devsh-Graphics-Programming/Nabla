#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLRenderBuffer.h"
#include "COpenGLDriver.h"

namespace irr
{
namespace video
{

//! constructor
COpenGLRenderBuffer::COpenGLRenderBuffer(GLenum internalFormat, core::dimension2du size, COpenGLDriver* driver)
    : InternalFormat(internalFormat), RenderBufferSize(size), Driver(driver), RenderBufferNameHasChanged(0), RenderBufferName(0)
{
    Driver->extGlCreateRenderbuffers(1,&RenderBufferName);
    Driver->extGlNamedRenderbufferStorage(RenderBufferName,InternalFormat,RenderBufferSize.Width,RenderBufferSize.Height);
}

//! destructor
COpenGLRenderBuffer::~COpenGLRenderBuffer()
{
    Driver->extGlDeleteRenderbuffers(1,&RenderBufferName);
}

void COpenGLRenderBuffer::resize(const core::dimension2du &newSize)
{
    RenderBufferSize = newSize;

    if (RenderBufferName)
    {
        glDeleteTextures(1,&RenderBufferName);
        RenderBufferName = 0;
    }
    Driver->extGlCreateRenderbuffers(1,&RenderBufferName);
    Driver->extGlNamedRenderbufferStorage(RenderBufferName,InternalFormat,RenderBufferSize.Width,RenderBufferSize.Height);

	RenderBufferNameHasChanged = CNullDriver::incrementAndFetchReallocCounter();
}

}
}
#endif
