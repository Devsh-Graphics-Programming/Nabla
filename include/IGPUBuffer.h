// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_GPU_BUFFER_H_INCLUDED__
#define __I_GPU_BUFFER_H_INCLUDED__

#include "IBuffer.h"
#include "IDriverMemoryBacked.h"

namespace irr
{
namespace video
{

//! GPU Buffer class, where the memory is provided by the driver, does not support resizing.
/** For additional OpenGL DSA state-free operations such as flushing mapped ranges or
buffer to buffer copies, one needs a command buffer in Vulkan as these operations are
performed by the GPU and not wholly by the driver, so look for them in IDriver and IVideoDriver. */
class IGPUBuffer : public core::IBuffer, public IDriverMemoryBacked
{
    protected:
        IGPUBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements& reqs) : IDriverMemoryBacked(reqs) {}

    public:
        //! Get usable buffer byte size.
        virtual const uint64_t& getSize() const {return cachedMemoryReqs.vulkanReqs.size;}

        //! Whether calling updateSubRange will produce any effects.
        virtual bool canUpdateSubRange() const = 0;

        //deprecated, delegate this to command buffer
        virtual void updateSubRange(const IDriverMemoryAllocation::MemoryRange& memrange, const void* data) = 0;


        //! A C++11 style move assignment, used for reallocation-like functionality.
        /** Only ICPUBuffer has a reallocate method, since IGPUBuffer needs an IDriverMemoryAllocation.
        In the future we may change this API, but the reallocation function would need to either take a
        memory allocator class and be able to free its current allocation from the old allocator.
        So if you can provide an already created and allocated IGPUBuffer, we can move its data members
        from `other` to `this`. However the method does not change the reference count of `other`,
        the `other` is simply made into an empty IGPUBuffer in a state similar to what it would be if it
        failed both creation and memory allocation binding. This is why I reccommend that `other` have a
        reference count of 1 and is not used or bound to any other resource such as a graphics pipeline
        or a texture buffer object. One can reallocate an IGPUBuffer like this:
        `{auto rep = Driver->createGPUBufferOnDedMem(newReqs, ... ); A->pseudoMoveAssign(rep); rep->drop();}`
        \returns true on success, method can fail for a number of reasons such as passing `other==this` .*/
        virtual bool pseudoMoveAssign(IGPUBuffer* other) = 0;
};

} // end namespace scene
} // end namespace irr

#endif

