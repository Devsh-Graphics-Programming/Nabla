#include "COpenGLPersistentlyMappedBuffer.h"

#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

COpenGLPersistentlyMappedBuffer::COpenGLPersistentlyMappedBuffer(const size_t &size, void* data, const GLbitfield &flags) : COpenGLBuffer(size,data,flags), persistentPointer(NULL)
{
    if (BufferName)
        return;

    persistentPointer = COpenGLExtensionHandler::extGlMapNamedBufferRange(BufferName,0,BufferSize,cachedFlags&(GL_MAP_WRITE_BIT|GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT));
}

COpenGLPersistentlyMappedBuffer::~COpenGLPersistentlyMappedBuffer()
{
    if (persistentPointer)
        COpenGLExtensionHandler::extGlUnmapNamedBuffer(BufferName);
}


bool COpenGLPersistentlyMappedBuffer::reallocate(const size_t &newSize, const bool& forceRetentionOfData, const bool &reallocateIfShrink)
{
    if (persistentPointer)
    {
        COpenGLExtensionHandler::extGlUnmapNamedBuffer(BufferName);
        persistentPointer = NULL;
    }

    bool success = COpenGLBuffer::reallocate(newSize,forceRetentionOfData,reallocateIfShrink);
    if (!success)
        return false;

    persistentPointer = COpenGLExtensionHandler::extGlMapNamedBufferRange(BufferName,0,BufferSize,cachedFlags&(GL_MAP_WRITE_BIT|GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT));

    return persistentPointer!=NULL;
}

} // end namespace video
} // end namespace irr

#endif
