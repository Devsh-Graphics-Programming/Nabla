#ifndef __C_OPEN_GL_BUFFER_H_INCLUDED__
#define __C_OPEN_GL_BUFFER_H_INCLUDED__

#include "IGPUBuffer.h"
#include "IrrCompileConfig.h"
#include "FW_Mutex.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
#include "CNullDriver.h"
#include "COpenGLExtensionHandler.h"
#include <assert.h>

namespace irr
{
namespace video
{



//! get the amount of Bits per Pixel of the given color format
inline uint32_t getBitsPerPixelFromGLenum(const GLenum& format)
{
    switch(format)
    {
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return 8;
        case GL_R16:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return 16;
        case GL_R32F:
        case GL_R32I:
        case GL_R32UI:
            return 32;
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
            return 16;
        case GL_RG16:
        case GL_RG16F:
        case GL_RG16I:
        case GL_RG16UI:
            return 32;
        case GL_RG32F:
        case GL_RG32I:
        case GL_RG32UI:
            return 64;
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return 96;
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
            return 32;
        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RGBA16I:
        case GL_RGBA16UI:
            return 64;
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return 128;
        default:
            return 0;
    }
}


class COpenGLBuffer : public IGPUBuffer, public IDriverMemoryAllocation
{
    protected:
        virtual ~COpenGLBuffer()
        {
            if (isCurrentlyMapped())
                unmapMemory();

#ifdef OPENGL_LEAK_DEBUG
            assert(concurrentAccessGuard==0);
            FW_AtomicCounterIncr(concurrentAccessGuard);
#endif // OPENGL_LEAK_DEBUG
            if (BufferName)
                COpenGLExtensionHandler::extGlDeleteBuffers(1,&BufferName);

#ifdef OPENGL_LEAK_DEBUG
            assert(concurrentAccessGuard==1);
            FW_AtomicCounterDecr(concurrentAccessGuard);
#endif // OPENGL_LEAK_DEBUG
        }

    public:
        COpenGLBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements &mreqs, const bool& canModifySubData) : IGPUBuffer(mreqs), BufferName(0), cachedFlags(0)
        {
			lastTimeReallocated = 0;
            COpenGLExtensionHandler::extGlCreateBuffers(1,&BufferName);
            if (BufferName==0)
                return;

            cachedFlags =   (canModifySubData ? GL_DYNAMIC_STORAGE_BIT:0)|
                            (mreqs.memoryHeapLocation==IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL ? GL_CLIENT_STORAGE_BIT:0);
            if (mreqs.mappingCapability&IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ)
                cachedFlags |= GL_MAP_PERSISTENT_BIT|GL_MAP_READ_BIT;
            if (mreqs.mappingCapability&IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE)
                cachedFlags |= GL_MAP_PERSISTENT_BIT|GL_MAP_WRITE_BIT;
            if (mreqs.mappingCapability&IDriverMemoryAllocation::EMCF_COHERENT)
                cachedFlags |= GL_MAP_COHERENT_BIT;
            COpenGLExtensionHandler::extGlNamedBufferStorage(BufferName,cachedMemoryReqs.vulkanReqs.size,nullptr,cachedFlags);

#ifdef OPENGL_LEAK_DEBUG
            for (size_t i=0; i<3; i++)
                concurrentAccessGuard[i] = 0;
#endif // OPENGL_LEAK_DEBUG
        }

        //!
        inline const GLuint& getOpenGLName() const {return BufferName;}


        //!
        virtual bool canUpdateSubRange() const {return cachedFlags&GL_DYNAMIC_STORAGE_BIT;}

        //!
        virtual void updateSubRange(const MemoryRange& memrange, const void* data)
        {
            if (canUpdateSubRange())
                COpenGLExtensionHandler::extGlNamedBufferSubData(BufferName,memrange.offset,memrange.length,data);
        }


        //! Returns the allocation which is bound to the resource
        virtual IDriverMemoryAllocation* getBoundMemory() {return this;}

        //! Constant version
        virtual const IDriverMemoryAllocation* getBoundMemory() const {return this;}

        //! Returns the offset in the allocation at which it is bound to the resource
        virtual size_t getBoundMemoryOffset() const {return 0ull;}


        //!
        virtual size_t getAllocationSize() const {return IGPUBuffer::getSize();}

        //!
        virtual E_SOURCE_MEMORY_TYPE getType() const {return ESMT_DONT_KNOW;}

        //!
        virtual E_MAPPING_CAPABILITY_FLAGS getMappingCaps() const {return static_cast<E_MAPPING_CAPABILITY_FLAGS>(cachedMemoryReqs.mappingCapability);} //move up?

        //!
        virtual void* mapMemoryRange(const E_MAPPING_CPU_ACCESS_FLAG& accessType, const MemoryRange& memrange)
        {
        #ifdef _DEBUG
            assert(!mappedPtr&&accessType!=EMCAF_NO_MAPPING_ACCESS&&BufferName);
            assert(accessType);
        #endif // _DEBUG
            GLbitfield flags = GL_MAP_PERSISTENT_BIT|((accessType&EMCAF_READ) ? GL_MAP_READ_BIT:0u);
            if (cachedFlags&GL_MAP_COHERENT_BIT)
            {
                flags |= GL_MAP_COHERENT_BIT|((accessType&EMCAF_WRITE) ? GL_MAP_WRITE_BIT:0u);
            }
            else if (accessType&EMCAF_WRITE)
            {
                flags |= GL_MAP_FLUSH_EXPLICIT_BIT|GL_MAP_WRITE_BIT;
            }
        #ifdef _DEBUG
            assert(((flags&(~cachedFlags))&(GL_MAP_READ_BIT|GL_MAP_WRITE_BIT))==0u);
        #endif // _DEBUG
            mappedPtr = reinterpret_cast<uint8_t*>(COpenGLExtensionHandler::extGlMapNamedBufferRange(BufferName,memrange.offset,memrange.length,flags))-memrange.offset;
            mappedRange = memrange;
            currentMappingAccess = static_cast<E_MAPPING_CPU_ACCESS_FLAG>(((flags&GL_MAP_READ_BIT) ? EMCAF_READ:0u)|((flags&GL_MAP_WRITE_BIT) ? EMCAF_WRITE:0u));
            return mappedPtr;
        }

        //!
        virtual void unmapMemory()
        {
        #ifdef _DEBUG
            assert(mappedPtr&&BufferName);
        #endif // _DEBUG
            COpenGLExtensionHandler::extGlUnmapNamedBuffer(BufferName);
            mappedPtr = nullptr;
            mappedRange = MemoryRange(0,0);
            currentMappingAccess = EMCAF_NO_MAPPING_ACCESS;
        }

        //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
        virtual bool isDedicated() const {return true;}
    protected:
        GLbitfield cachedFlags;
        GLuint BufferName;

        virtual bool pseudoMoveAssign(IGPUBuffer* other)
        {
            COpenGLBuffer* otherAsGL = static_cast<COpenGLBuffer*>(other);
            if (!otherAsGL || otherAsGL==this || otherAsGL->cachedFlags!=cachedFlags || otherAsGL->BufferName==0)
                return false;

            #ifdef _DEBUG
            if (otherAsGL->getReferenceCount()!=1)
                os::Printer::log("What are you doing!? You should only swap internals with an IGPUBuffer that is unused yet!",ELL_ERROR);
            #endif // _DEBUG

            #ifdef OPENGL_LEAK_DEBUG
                assert(otherAsGL->concurrentAccessGuard==0);
                FW_AtomicCounterIncr(otherAsGL->concurrentAccessGuard);
                assert(concurrentAccessGuard==0);
                FW_AtomicCounterIncr(concurrentAccessGuard);
            #endif // OPENGL_LEAK_DEBUG

            if (BufferName)
                COpenGLExtensionHandler::extGlDeleteBuffers(1,&BufferName);

            cachedMemoryReqs = otherAsGL->cachedMemoryReqs;

            cachedFlags = otherAsGL->cachedFlags;
            BufferName = otherAsGL->BufferName;

            lastTimeReallocated = CNullDriver::incrementAndFetchReallocCounter();


            otherAsGL->cachedMemoryReqs = {{0,0,0},0,0,0,0};

            otherAsGL->cachedFlags = 0;
            otherAsGL->BufferName = 0;

            otherAsGL->lastTimeReallocated = CNullDriver::incrementAndFetchReallocCounter();

            #ifdef OPENGL_LEAK_DEBUG
                assert(concurrentAccessGuard==1);
                FW_AtomicCounterDecr(concurrentAccessGuard);
                assert(otherAsGL->concurrentAccessGuard==1);
                FW_AtomicCounterDecr(otherAsGL->concurrentAccessGuard);
            #endif // OPENGL_LEAK_DEBUG

            return true;
        }

    private:
#ifdef OPENGL_LEAK_DEBUG
        FW_AtomicCounter concurrentAccessGuard;
#endif // OPENGL_LEAK_DEBUG
};

} // end namespace video
} // end namespace irr

#endif
#endif
