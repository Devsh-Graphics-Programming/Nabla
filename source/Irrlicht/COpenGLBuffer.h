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

class COpenGLBuffer final : public IGPUBuffer, public IDriverMemoryAllocation
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
        inline bool canUpdateSubRange() const override {return cachedFlags&GL_DYNAMIC_STORAGE_BIT;}

        //!
        inline void updateSubRange(const MemoryRange& memrange, const void* data) override
        {
            if (canUpdateSubRange())
                COpenGLExtensionHandler::extGlNamedBufferSubData(BufferName,memrange.offset,memrange.length,data);
        }


        //! Returns the allocation which is bound to the resource
        inline IDriverMemoryAllocation* getBoundMemory() override {return this;}

        //! Constant version
        inline const IDriverMemoryAllocation* getBoundMemory() const override {return this;}

        //! Returns the offset in the allocation at which it is bound to the resource
        inline size_t getBoundMemoryOffset() const override {return 0ull;}


        //!
        inline size_t getAllocationSize() const override {return IGPUBuffer::getSize();}

        //!
        inline E_SOURCE_MEMORY_TYPE getType() const override {return ESMT_DONT_KNOW;}

        //!
        inline E_MAPPING_CAPABILITY_FLAGS getMappingCaps() const override {return static_cast<E_MAPPING_CAPABILITY_FLAGS>(cachedMemoryReqs.mappingCapability);} //move up?

        //!
        inline void* mapMemoryRange(const E_MAPPING_CPU_ACCESS_FLAG& accessType, const MemoryRange& memrange) override
        {
        #ifdef _DEBUG
            assert(!mappedPtr&&accessType!=EMCAF_NO_MAPPING_ACCESS&&BufferName);
            assert(accessType);
        #endif // _DEBUG

            GLbitfield flags = GL_MAP_PERSISTENT_BIT|(accessType&static_cast<GLbitfield>(GL_MAP_READ_BIT) ? GL_MAP_READ_BIT:0u);
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
            bool canRead = flags&static_cast<GLbitfield>(GL_MAP_READ_BIT);
            bool canWrite = flags&static_cast<GLbitfield>(GL_MAP_WRITE_BIT);
            currentMappingAccess = static_cast<E_MAPPING_CPU_ACCESS_FLAG>((canRead ? static_cast<uint32_t>(EMCAF_READ):0u)|(canWrite ? static_cast<uint32_t>(EMCAF_WRITE):0u));
            return mappedPtr;
        }

        //!
        inline void unmapMemory() override
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
        inline bool isDedicated() const override {return true;}
    protected:
        GLbitfield cachedFlags;
        GLuint BufferName;

        inline bool pseudoMoveAssign(IGPUBuffer* other) override
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

            mappedPtr = otherAsGL->mappedPtr;
            mappedRange = otherAsGL->mappedRange;
            currentMappingAccess = otherAsGL->currentMappingAccess;

            cachedFlags = otherAsGL->cachedFlags;
            BufferName = otherAsGL->BufferName;

            lastTimeReallocated = CNullDriver::incrementAndFetchReallocCounter();


            otherAsGL->cachedMemoryReqs = {{0,0,0},0,0,0,0};

            otherAsGL->mappedPtr = nullptr;
            otherAsGL->mappedRange = MemoryRange(0,0);
            otherAsGL->currentMappingAccess = EMCAF_NO_MAPPING_ACCESS;

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
