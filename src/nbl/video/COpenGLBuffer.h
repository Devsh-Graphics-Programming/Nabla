// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPEN_GL_BUFFER_H_INCLUDED__
#define __NBL_C_OPEN_GL_BUFFER_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/video/IGPUBuffer.h"

#include <assert.h>

#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl::video
{

class COpenGLBuffer final : public IGPUBuffer, public IDriverMemoryAllocation
{
    protected:
        virtual ~COpenGLBuffer();

    public:
        COpenGLBuffer(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev, IOpenGL_FunctionTable* gl,
            const IDriverMemoryBacked::SDriverMemoryRequirements &mreqs,
            const IGPUBuffer::SCachedCreationParams& cachedCreationParams
        ) : IGPUBuffer(std::move(dev),mreqs,cachedCreationParams), IDriverMemoryAllocation(getOriginDevice()), BufferName(0), cachedFlags(0)
        {
            gl->extGlCreateBuffers(1,&BufferName);
            if (BufferName==0)
                return;

            cachedFlags =   (cachedCreationParams.canUpdateSubRange ? GL_DYNAMIC_STORAGE_BIT:0)|
                            (mreqs.memoryHeapLocation==IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL ? GL_CLIENT_STORAGE_BIT:0);
            if (mreqs.mappingCapability&IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ)
                cachedFlags |= GL_MAP_PERSISTENT_BIT|GL_MAP_READ_BIT;
            if (mreqs.mappingCapability&IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE)
                cachedFlags |= GL_MAP_PERSISTENT_BIT|GL_MAP_WRITE_BIT;
            if (mreqs.mappingCapability&IDriverMemoryAllocation::EMCF_COHERENT)
                cachedFlags |= GL_MAP_COHERENT_BIT;
            gl->extGlNamedBufferStorage(BufferName,cachedMemoryReqs.vulkanReqs.size,nullptr,cachedFlags);
        }

        void setObjectDebugName(const char* label) const override;

        //!
        inline const GLuint& getOpenGLName() const {return BufferName;}

        //!
        inline GLbitfield getOpenGLStorageFlags() const { return cachedFlags; }

        //! Returns the allocation which is bound to the resource
        inline IDriverMemoryAllocation* getBoundMemory() override {return this;}

        //! Constant version
        inline const IDriverMemoryAllocation* getBoundMemory() const override {return this;}

        //! Returns the offset in the allocation at which it is bound to the resource
        inline size_t getBoundMemoryOffset() const override {return 0ull;}

        //! on OpenGL the buffer is the allocation
        inline size_t getAllocationSize() const override {return IGPUBuffer::getSize();}

        //!
        inline E_SOURCE_MEMORY_TYPE getType() const override {return ESMT_DONT_KNOW;}

        //!
        inline E_MAPPING_CAPABILITY_FLAGS getMappingCaps() const override {return static_cast<E_MAPPING_CAPABILITY_FLAGS>(cachedMemoryReqs.mappingCapability);} //move up?

        //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
        inline bool isDedicated() const override { return true; }

        inline uint64_t getBufferSize() const override { return getSize(); }

    protected:
        GLbitfield cachedFlags;
        GLuint BufferName;
};

} // end namespace nbl::video

#endif
