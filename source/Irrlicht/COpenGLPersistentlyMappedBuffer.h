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
        COpenGLPersistentlyMappedBuffer(const size_t &size, void* data, const GLbitfield &flags);
        virtual ~COpenGLPersistentlyMappedBuffer();

        virtual bool reallocate(const size_t &newSize, const bool& forceRetentionOfData=false, const bool &reallocateIfShrink=false);

        //! WARNING: RESIZE will invalidate pointer
        //! WARNING: NEED TO FENCE BEFORE USE!!!!!!!!!!!!!
        virtual void* getPointer() {return persistentPointer;}
    private:
        void* persistentPointer;
};

} // end namespace video
} // end namespace irr

#endif
#endif
