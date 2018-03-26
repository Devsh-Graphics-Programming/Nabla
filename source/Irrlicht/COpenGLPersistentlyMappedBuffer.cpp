#include "COpenGLPersistentlyMappedBuffer.h"

#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

COpenGLPersistentlyMappedBuffer::COpenGLPersistentlyMappedBuffer(const size_t &size, const void* data, const GLbitfield &flags, const GLbitfield &mapOnCreation_andFlags) : COpenGLBuffer(size,data,flags), persistentPointer(NULL)
{
    if (!BufferName)
        return;

    if (mapOnCreation_andFlags&GL_MAP_WRITE_BIT)
        cachedMappingFlags = mapOnCreation_andFlags&(GL_MAP_WRITE_BIT|GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT|GL_MAP_FLUSH_EXPLICIT_BIT)|(mapOnCreation_andFlags&GL_MAP_READ_BIT ? 0:GL_MAP_INVALIDATE_BUFFER_BIT);
    else if (mapOnCreation_andFlags&GL_MAP_READ_BIT)
        cachedMappingFlags = mapOnCreation_andFlags&(GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT);
    else
    {
        cachedMappingFlags = 0;
        return;
    }

    persistentPointer = COpenGLExtensionHandler::extGlMapNamedBufferRange(BufferName,0,BufferSize,cachedMappingFlags);
}

COpenGLPersistentlyMappedBuffer::~COpenGLPersistentlyMappedBuffer()
{
    if (persistentPointer)
        COpenGLExtensionHandler::extGlUnmapNamedBuffer(BufferName);
}


bool COpenGLPersistentlyMappedBuffer::reallocate(const size_t &newSize, const bool& forceRetentionOfData, const bool &reallocateIfShrink, const size_t& wraparoundStart)
{
    GLbitfield flags = cachedMappingFlags;
    if (persistentPointer)
    {
        COpenGLExtensionHandler::extGlUnmapNamedBuffer(BufferName);
        persistentPointer = NULL;
        cachedMappingFlags = 0;
    }

    bool success = COpenGLBuffer::reallocate(newSize,forceRetentionOfData,reallocateIfShrink, wraparoundStart);
    if (!success)
        return false;

    MapBufferRange(flags,0,newSize);
    return true;
}

} // end namespace video
} // end namespace irr

#endif
