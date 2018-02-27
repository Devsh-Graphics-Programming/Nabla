#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLDriver.h"

#include "COpenGLRenderBuffer.h"

namespace irr
{
namespace video
{

//! constructor
COpenGLRenderBuffer::COpenGLRenderBuffer(GLenum internalFormat, core::dimension2du size)
    : RenderBufferSize(size), InternalFormat(internalFormat), RenderBufferName(0), RenderBufferNameHasChanged(0)
{
    COpenGLExtensionHandler::extGlCreateRenderbuffers(1,&RenderBufferName);
    COpenGLExtensionHandler::extGlNamedRenderbufferStorage(RenderBufferName,InternalFormat,RenderBufferSize.Width,RenderBufferSize.Height);
}

//! destructor
COpenGLRenderBuffer::~COpenGLRenderBuffer()
{
    COpenGLExtensionHandler::extGlDeleteRenderbuffers(1,&RenderBufferName);
}

void COpenGLRenderBuffer::resize(const core::dimension2du &newSize)
{
    RenderBufferSize = newSize;

    if (RenderBufferName)
    {
        COpenGLExtensionHandler::extGlDeleteRenderbuffers(1,&RenderBufferName);
        RenderBufferName = 0;
    }
    COpenGLExtensionHandler::extGlCreateRenderbuffers(1,&RenderBufferName);
    COpenGLExtensionHandler::extGlNamedRenderbufferStorage(RenderBufferName,InternalFormat,RenderBufferSize.Width,RenderBufferSize.Height);

	RenderBufferNameHasChanged = CNullDriver::incrementAndFetchReallocCounter();
}




//! constructor
COpenGLMultisampleRenderBuffer::COpenGLMultisampleRenderBuffer(GLenum internalFormat, core::dimension2du size, uint32_t sampleCount)
    : COpenGLRenderBuffer(internalFormat, size, false), SampleCount(sampleCount)
{
    COpenGLExtensionHandler::extGlCreateRenderbuffers(1,&RenderBufferName);
    COpenGLExtensionHandler::extGlNamedRenderbufferStorageMultisample(RenderBufferName,SampleCount,InternalFormat,RenderBufferSize.Width,RenderBufferSize.Height);
}


void COpenGLMultisampleRenderBuffer::resize(const core::dimension2du &newSize)
{
    RenderBufferSize = newSize;

    if (RenderBufferName)
    {
        COpenGLExtensionHandler::extGlDeleteRenderbuffers(1,&RenderBufferName);
        RenderBufferName = 0;
    }
    COpenGLExtensionHandler::extGlCreateRenderbuffers(1,&RenderBufferName);
    COpenGLExtensionHandler::extGlNamedRenderbufferStorageMultisample(RenderBufferName,SampleCount,InternalFormat,RenderBufferSize.Width,RenderBufferSize.Height);

	RenderBufferNameHasChanged = CNullDriver::incrementAndFetchReallocCounter();
}

}
}
#endif
