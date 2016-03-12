#ifndef __C_OPEN_GL_BUFFER_H_INCLUDED__
#define __C_OPEN_GL_BUFFER_H_INCLUDED__

#include "IGPUBuffer.h"
#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
#include "COpenGLExtensionHandler.h"

namespace irr
{
namespace video
{

class COpenGLBuffer : public virtual video::IGPUBuffer
{
    public:
        COpenGLBuffer(const size_t &size, void* data, const GLbitfield &flags) : BufferName(0), BufferSize(0), cachedFlags(0)
        {
            COpenGLExtensionHandler::extGlCreateBuffers(1,&BufferName);
            if (BufferName==0)
                return;

            COpenGLExtensionHandler::extGlNamedBufferStorage(BufferName,size,data,flags);
            cachedFlags = flags;
            BufferSize = size;
        }

        virtual ~COpenGLBuffer()
        {
            if (BufferName)
                COpenGLExtensionHandler::extGlDeleteBuffers(1,&BufferName);
        }


        virtual core::E_BUFFER_TYPE getBufferType() const {return core::EBT_UNSPECIFIED_BUFFER;}

        virtual const GLuint& getOpenGLName() const {return BufferName;}

        virtual const uint64_t &getSize() const {return BufferSize;}

        virtual void updateSubRange(const size_t& offset, const size_t& size, void* data)
        {
            if (cachedFlags&GL_DYNAMIC_STORAGE_BIT)
                COpenGLExtensionHandler::extGlNamedBufferSubData(BufferName,offset,size,data);
        }

        virtual bool canUpdateSubRange() const {return cachedFlags&GL_DYNAMIC_STORAGE_BIT;}

        virtual bool reallocate(const size_t &newSize, const bool& forceRetentionOfData=false, const bool &reallocateIfShrink=false)
        {
            if (newSize==BufferSize)
                return true;

            if (newSize<BufferSize&&(!reallocateIfShrink))
                return true;

            if (forceRetentionOfData)
            {
                GLuint newBufferHandle = 0;
                COpenGLExtensionHandler::extGlCreateBuffers(1,&newBufferHandle);
                if (newBufferHandle==0)
                    return false;

                COpenGLExtensionHandler::extGlNamedBufferStorage(newBufferHandle,newSize,NULL,cachedFlags);
                COpenGLExtensionHandler::extGlCopyNamedBufferSubData(BufferName,newBufferHandle,0,0,core::min_(newSize,BufferSize));
                BufferSize = newSize;

                COpenGLExtensionHandler::extGlDeleteBuffers(1,&BufferName);
                BufferName = newBufferHandle;
            }
            else
            {
                COpenGLExtensionHandler::extGlDeleteBuffers(1,&BufferName);
                COpenGLExtensionHandler::extGlCreateBuffers(1,&BufferName);
                if (BufferName==0)
                    return false;

                COpenGLExtensionHandler::extGlNamedBufferStorage(BufferName,newSize,NULL,cachedFlags);
                BufferSize = newSize;
            }
            lastTimeReallocated = os::Timer::getRealTime();

            return true;
        }
    protected:
        GLbitfield cachedFlags;
        size_t BufferSize;
        GLuint BufferName;
};

} // end namespace video
} // end namespace irr

#endif
#endif
