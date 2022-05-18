// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPEN_GL_BUFFER_H_INCLUDED__
#define __NBL_C_OPEN_GL_BUFFER_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/IOpenGLMemoryAllocation.h"

#include <assert.h>

#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl::video
{

class COpenGLBuffer final : public IGPUBuffer, public IOpenGLMemoryAllocation
{
    protected:
        virtual ~COpenGLBuffer();

    public:
        COpenGLBuffer(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev, IOpenGL_FunctionTable* gl,
            const IDriverMemoryBacked::SDriverMemoryRequirements &mreqs,
            const IGPUBuffer::SCachedCreationParams& cachedCreationParams
        ) : IGPUBuffer(std::move(dev),mreqs,cachedCreationParams), IOpenGLMemoryAllocation(getOriginDevice()), BufferName(0), cachedFlags(0)
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
        COpenGLBuffer(
            core::smart_refctd_ptr<const ILogicalDevice>&& dev,
            const IDriverMemoryBacked::SDriverMemoryRequirements2 &mreqs,
            const IGPUBuffer::SCachedCreationParams& cachedCreationParams,
            GLuint bufferName
        ) : IGPUBuffer(std::move(dev),mreqs,cachedCreationParams), IOpenGLMemoryAllocation(getOriginDevice()), BufferName(bufferName), cachedFlags(0)
        {
        }

        bool initMemory(
            IOpenGL_FunctionTable* gl,
            core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
            core::bitflag<IDriverMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags) override
        {
            if(!IOpenGLMemoryAllocation::initMemory(gl, allocateFlags, memoryPropertyFlags))
                return false;
            cachedFlags =   (m_cachedCreationParams.canUpdateSubRange ? GL_DYNAMIC_STORAGE_BIT:0)|
                            (memoryPropertyFlags.hasFlags(IDriverMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) ? 0:GL_CLIENT_STORAGE_BIT);
            if (memoryPropertyFlags.hasFlags(IDriverMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_READABLE_BIT))
                cachedFlags |= GL_MAP_PERSISTENT_BIT|GL_MAP_READ_BIT;
            if (memoryPropertyFlags.hasFlags(IDriverMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_WRITABLE_BIT))
                cachedFlags |= GL_MAP_PERSISTENT_BIT|GL_MAP_WRITE_BIT;
            if (memoryPropertyFlags.hasFlags(IDriverMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_COHERENT_BIT))
                cachedFlags |= GL_MAP_COHERENT_BIT;
            gl->extGlNamedBufferStorage(BufferName,cachedMemoryReqs2.size,nullptr,cachedFlags);
        }

        void setObjectDebugName(const char* label) const override;

        //!
        inline const void* getNativeHandle() const override {return &BufferName;}
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

        //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
        inline bool isDedicated() const override { return true; }

    protected:
        GLbitfield cachedFlags;
        GLuint BufferName;
};

} // end namespace nbl::video

#endif
