// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED_
#define _NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED_

#include "IDeviceMemoryAllocation.h"
#include <algorithm>

namespace nbl::video
{

//! Interface from which resources backed by IDeviceMemoryAllocation inherit from
class IDeviceMemoryBacked : public virtual core::IReferenceCounted
{
    public:
        //! We need to know to cast to `IGPUBuffer` or `IGPUImage`
        enum E_OBJECT_TYPE : bool
        {
            EOT_BUFFER,
            EOT_IMAGE
        };

        //! For allocating from Heaps exposed to Nabla
        struct SDeviceMemoryRequirements
        {
            // the allocation size required to back the resource
            size_t   size; // TODO: C++23 default-initialize to 0ull
            // a bitmask of all memory type IDs (one bit per ID) which are compatible with the resource
            uint32_t memoryTypeBits; // TODO: C++23 default-initialize to 0x0u
            // what alignment should be memory allocation should have, encoded as Log2 as alignments need to be PoT
            uint32_t alignmentLog2 : 6; // TODO: C++23 default-initialize to 63
            // whether you'll get better performance from having one allocation exclusively bound to this resource
            uint32_t prefersDedicatedAllocation : 1; // TODO: C++23 default-initialize to true
            // whether you need to have one allocation exclusively bound to this resource, always true in OpenGL
            uint32_t requiresDedicatedAllocation : 1; // TODO: C++23 default-initialize to true
            // Whether the destructor will skip the call to `vkDestory` or `glDelete` on the handle, this is only useful for "imported" images
            uint32_t merelyObservesHandle : 1; // TODO: C++23 default-initialize to false
        };
        static_assert(sizeof(SDeviceMemoryRequirements)==16);

        //! If you bound an "exotic" memory object to the resource, you might require "special" cleanups in the destructor
        struct ICleanup
        {
            virtual ~ICleanup() = 0;
        };
        

        //! Return type of memory backed object (image or buffer)
        virtual E_OBJECT_TYPE getObjectType() const = 0;

        //! Before allocating memory from the driver or trying to bind a range of an existing allocation
        inline const SDeviceMemoryRequirements& getMemoryReqs() const {return cachedMemoryReqs;}

        //! Returns the allocation which is bound to the resource
        virtual IDeviceMemoryAllocation* getBoundMemory() = 0;

        //! Constant version
        virtual const IDeviceMemoryAllocation* getBoundMemory() const = 0;

        //! Returns the offset in the allocation at which it is bound to the resource
        virtual size_t getBoundMemoryOffset() const = 0;

    protected:
        inline IDeviceMemoryBacked(std::unique_ptr<ICleanup>&& _preStep, const SDeviceMemoryRequirements& reqs, std::unique_ptr<ICleanup>&& _postStep)
            : preStep(std::move(_preStep)), cachedMemoryReqs(reqs), postStep(std::move(_postStep)) {}
        inline virtual ~IDeviceMemoryBacked()
        {
            assert(!preStep); // derived class should have already cleared this out
        }

        // it's the derived class' responsibility to call this in its destructor
        inline void preDestroyStep()
        {
            preStep = nullptr;
        }

        //! members
        std::unique_ptr<ICleanup> preStep;
        SDeviceMemoryRequirements cachedMemoryReqs;
        std::unique_ptr<ICleanup> postStep;
};

} // end namespace nbl::video

#endif
