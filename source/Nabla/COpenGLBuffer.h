// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPEN_GL_BUFFER_H_INCLUDED__
#define __NBL_C_OPEN_GL_BUFFER_H_INCLUDED__

#include "nbl/core/core.h"
#include "IGPUBuffer.h"

#include "FW_Mutex.h"

#include "nbl/video/IOpenGL_FunctionTable.h"
#include <assert.h>
#include <atomic>

namespace nbl
{
namespace video
{

class COpenGLBuffer final : public IGPUBuffer, public IDriverMemoryAllocation
{
    protected:
        virtual ~COpenGLBuffer()
        {
            destroyGLBufferObjectWrapper();
        }

        void destroyGLBufferObjectWrapper();

    public:
        COpenGLBuffer(ILogicalDevice* dev, IOpenGL_FunctionTable* gl, const IDriverMemoryBacked::SDriverMemoryRequirements &mreqs, const bool& canModifySubData) : IGPUBuffer(dev, mreqs), BufferName(0), cachedFlags(0)
        {
			lastTimeReallocated = 0;
            gl->extGlCreateBuffers(1,&BufferName);
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
            gl->extGlNamedBufferStorage(BufferName,cachedMemoryReqs.vulkanReqs.size,nullptr,cachedFlags);
        }

        //!
        inline const GLuint& getOpenGLName() const {return BufferName;}

        //!
        inline GLbitfield getOpenGLStorageFlags() const { return cachedFlags; }

        //!
        inline bool canUpdateSubRange() const override {return cachedFlags&IOpenGL_FunctionTable::DYNAMIC_STORAGE_BIT;}

        //!
        /*
        inline void updateSubRange(const MemoryRange& memrange, const void* data) override
        {
            if (canUpdateSubRange())
                COpenGLExtensionHandler::extGlNamedBufferSubData(BufferName,memrange.offset,memrange.length,data);
        }
        */

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

        //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
        inline bool isDedicated() const override { return true; }

        inline bool pseudoMoveAssign(IGPUBuffer* other) override
        {
            if (!this->isCompatibleDevicewise(other))
                return false;

            COpenGLBuffer* otherAsGL = static_cast<COpenGLBuffer*>(other);
            if (!otherAsGL || otherAsGL == this || otherAsGL->cachedFlags != cachedFlags || otherAsGL->BufferName == 0)
                return false;

#ifdef _DEBUG
            if (otherAsGL->getReferenceCount() != 1)
                os::Printer::log("What are you doing!? You should only swap internals with an IGPUBuffer that is unused yet!", ELL_ERROR);
#endif // _DEBUG

#ifdef OPENGL_LEAK_DEBUG
            assert(otherAsGL->concurrentAccessGuard == 0);
            FW_AtomicCounterIncr(otherAsGL->concurrentAccessGuard);
            assert(concurrentAccessGuard == 0);
            FW_AtomicCounterIncr(concurrentAccessGuard);
#endif // OPENGL_LEAK_DEBUG

            if (BufferName)
                destroyGLBufferObjectWrapper();

            cachedMemoryReqs = otherAsGL->cachedMemoryReqs;

            mappedPtr = otherAsGL->mappedPtr;
            mappedRange = otherAsGL->mappedRange;
            currentMappingAccess = otherAsGL->currentMappingAccess;

            cachedFlags = otherAsGL->cachedFlags;
            BufferName = otherAsGL->BufferName;

            lastTimeReallocated = s_reallocCounter++;


            otherAsGL->cachedMemoryReqs = { {0,0,0},0,0,0,0 };

            otherAsGL->mappedPtr = nullptr;
            otherAsGL->mappedRange = MemoryRange(0, 0);
            otherAsGL->currentMappingAccess = EMCAF_NO_MAPPING_ACCESS;

            otherAsGL->cachedFlags = 0;
            otherAsGL->BufferName = 0;

            otherAsGL->lastTimeReallocated = s_reallocCounter++;

#ifdef OPENGL_LEAK_DEBUG
            assert(concurrentAccessGuard == 1);
            FW_AtomicCounterDecr(concurrentAccessGuard);
            assert(otherAsGL->concurrentAccessGuard == 1);
            FW_AtomicCounterDecr(otherAsGL->concurrentAccessGuard);
#endif // OPENGL_LEAK_DEBUG

            return true;
        }

    protected:
        static std::atomic_uint32_t s_reallocCounter;

        GLbitfield cachedFlags;
        GLuint BufferName;
};

} // end namespace video
} // end namespace nbl

#endif
