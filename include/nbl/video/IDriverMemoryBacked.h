// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED__
#define __NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED__

#include "IDriverMemoryAllocation.h"
#include <algorithm>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace nbl::video
{

//! Interface from which resources backed by IDriverMemoryAllocation, such as ITexture and IGPUBuffer, inherit from
class IDriverMemoryBacked : public virtual core::IReferenceCounted
{
    public:
        enum E_OBJECT_TYPE : bool
        {
            EOT_BUFFER,
            EOT_IMAGE
        };

        struct SDriverMemoryRequirements
        {
            size_t   size;
            uint32_t memoryTypeBits;
            uint32_t alignmentLog2 : 6;
            uint32_t prefersDedicatedAllocation     : 1;
            uint32_t requiresDedicatedAllocation    : 1;
        };
        static_assert(sizeof(SDriverMemoryRequirements)==16);
        
        //! Return type of memory backed object (image or buffer)
        virtual E_OBJECT_TYPE getObjectType() const = 0;

        //! Before allocating memory from the driver or trying to bind a range of an existing allocation
        inline const SDriverMemoryRequirements& getMemoryReqs2() const {return cachedMemoryReqs2;}

        //! Returns the allocation which is bound to the resource
        virtual IDriverMemoryAllocation* getBoundMemory() = 0;

        //! Constant version
        virtual const IDriverMemoryAllocation* getBoundMemory() const = 0;

        //! Returns the offset in the allocation at which it is bound to the resource
        virtual size_t getBoundMemoryOffset() const = 0;

    protected:
        IDriverMemoryBacked() {}
        IDriverMemoryBacked(const SDriverMemoryRequirements& reqs) : cachedMemoryReqs2(reqs) {}

        SDriverMemoryRequirements cachedMemoryReqs2;
};

} // end namespace nbl::video

#endif
