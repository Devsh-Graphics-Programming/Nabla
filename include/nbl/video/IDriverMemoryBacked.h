// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED__
#define __NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED__

#include "IDriverMemoryAllocation.h"
#include <algorithm>

namespace nbl::video
{

// TODO: get rid of double defs
typedef uint64_t VkDeviceSize;
//placeholder until we configure Vulkan SDK
typedef struct VkMemoryRequirements {
    VkDeviceSize    size;
    VkDeviceSize    alignment; /// Used and valid only in Vulkan
    uint32_t        memoryTypeBits; /// Used and valid only in Vulkan
} VkMemoryRequirements; //depr

//! Interface from which resources backed by IDriverMemoryAllocation, such as ITexture and IGPUBuffer, inherit from
class IDriverMemoryBacked : public virtual core::IReferenceCounted
{
    public:
        struct SDriverMemoryRequirements
        {
            VkMemoryRequirements vulkanReqs;
            uint32_t memoryHeapLocation             : 2; //IDriverMemoryAllocation::E_SOURCE_MEMORY_TYPE
            uint32_t mappingCapability              : 4; //IDriverMemoryAllocation::E_MAPPING_CAPABILITY_FLAGS
            uint32_t prefersDedicatedAllocation     : 1; /// Used and valid only in Vulkan
            uint32_t requiresDedicatedAllocation    : 1; /// Used and valid only in Vulkan
        };
        //! Combine requirements
        /** \return true on success, some requirements are mutually exclusive, so it may be impossible to combine them. */
        static inline bool combineRequirements(SDriverMemoryRequirements& out, const SDriverMemoryRequirements& a, const SDriverMemoryRequirements& b)
        {
            switch (a.memoryHeapLocation)
            {
                case IDriverMemoryAllocation::ESMT_DEVICE_LOCAL:
					if (b.memoryHeapLocation==IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL)
						return false;
					out.memoryHeapLocation = a.memoryHeapLocation;
					break;
                case IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL:
                    if (b.memoryHeapLocation==IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
                        return false;
                    out.memoryHeapLocation = a.memoryHeapLocation;
                    break;
                default:
                    out.memoryHeapLocation = b.memoryHeapLocation;
                    break;
            }
            out.mappingCapability = a.mappingCapability|b.mappingCapability;
            out.prefersDedicatedAllocation = a.prefersDedicatedAllocation|b.prefersDedicatedAllocation;
            out.requiresDedicatedAllocation = a.requiresDedicatedAllocation|b.requiresDedicatedAllocation;

            //! Not on Vulkan, then OpenGL doesn't need more checks
            if (a.vulkanReqs.alignment==0u&&b.vulkanReqs.alignment==0u)
                return true;

            //! On Vulkan and don't know is not an option [can be removed later]
            if (out.memoryHeapLocation==IDriverMemoryAllocation::ESMT_DONT_KNOW)
                return false;

            auto isPowerOfTwo = [] (const uint64_t& N) -> bool {return N && !(N & (N - 1ll));};
            if (!isPowerOfTwo(a.vulkanReqs.alignment) || !isPowerOfTwo(b.vulkanReqs.alignment))
                return false;

            out.vulkanReqs.size = std::max<uint32_t>(a.vulkanReqs.size,b.vulkanReqs.size);
            out.vulkanReqs.alignment = std::max<uint32_t>(a.vulkanReqs.alignment,b.vulkanReqs.alignment);
            out.vulkanReqs.memoryTypeBits = a.vulkanReqs.memoryTypeBits&b.vulkanReqs.memoryTypeBits;
            if (out.vulkanReqs.memoryTypeBits!=0u)
                return false;

            return true;
        }

        //! Before allocating memory from the driver or trying to bind a range of an existing allocation
        inline const SDriverMemoryRequirements& getMemoryReqs() const {return cachedMemoryReqs;}

        //! Returns the allocation which is bound to the resource
        virtual IDriverMemoryAllocation* getBoundMemory() = 0;

        //! Constant version
        virtual const IDriverMemoryAllocation* getBoundMemory() const = 0;

        //! Returns the offset in the allocation at which it is bound to the resource
        virtual size_t getBoundMemoryOffset() const = 0;

    protected:
        IDriverMemoryBacked() {}
        IDriverMemoryBacked(const SDriverMemoryRequirements& reqs) : cachedMemoryReqs(reqs) {}

        SDriverMemoryRequirements cachedMemoryReqs;
        // TODO: backward link to the IDriverMemoryAllocation
};

} // end namespace nbl::video

#endif
