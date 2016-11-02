#ifndef __C_OPEN_GL_PERSISTENTLY_MAPPED_BUFFER_H_INCLUDED__
#define __C_OPEN_GL_PERSISTENTLY_MAPPED_BUFFER_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "IGPUMappedBuffer.h"
#include "COpenGLBuffer.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

class COpenGLPersistentlyMappedBuffer : public IGPUMappedBuffer, COpenGLBuffer
{
    public:
        COpenGLPersistentlyMappedBuffer(const size_t &size, const void* data, const GLbitfield &flags, const GLbitfield &mapOnCreation_andFlags);
        virtual ~COpenGLPersistentlyMappedBuffer();

        virtual bool reallocate(const size_t &newSize, const bool& forceRetentionOfData=false, const bool &reallocateIfShrink=false);

        virtual const GLuint& getOpenGLName() const {return BufferName;}

        inline void* MapBufferRange(const GLbitfield &mapFlags, const size_t& start, const size_t& length)
        {
            if (mapFlags&GL_MAP_WRITE_BIT)
            {
                if (mapFlags&GL_MAP_READ_BIT)
                    cachedMappingFlags = mapFlags&(GL_MAP_WRITE_BIT|GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT);
                else
                    cachedMappingFlags = mapFlags&(GL_MAP_WRITE_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT|GL_MAP_FLUSH_EXPLICIT_BIT|GL_MAP_INVALIDATE_RANGE_BIT);
            }
            else if (mapFlags&GL_MAP_READ_BIT)
                cachedMappingFlags = mapFlags&(GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT);
            else
            {
                cachedMappingFlags = 0;
                return NULL;
            }

            persistentPointer = COpenGLExtensionHandler::extGlMapNamedBufferRange(BufferName,start,length,cachedMappingFlags);
            return persistentPointer;
        }

        inline void Unmap()
        {
            if (persistentPointer)
            {
                COpenGLExtensionHandler::extGlUnmapNamedBuffer(BufferName);
                cachedMappingFlags = 0;
                persistentPointer = NULL;
            }
        }


        //! WARNING: RESIZE will invalidate pointer
        //! WARNING: NEED TO FENCE BEFORE USE!!!!!!!!!!!!!
        virtual void* getPointer() {return persistentPointer;}


    protected:
        virtual bool reallocate(const size_t &newSize, const bool& forceRetentionOfData, const bool &reallocateIfShrink, const size_t& wraparoundStart);

    private:
        void* persistentPointer;
        GLbitfield cachedMappingFlags;
};

} // end namespace video
} // end namespace irr

#endif
#endif
