// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED__
#define __NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED__

#include "IDeviceMemoryAllocation.h"
#include <algorithm>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace nbl::video
{

//! Interface from which resources backed by IDeviceMemoryAllocation, such as ITexture and IGPUBuffer, inherit from
class NBL_API IDriverMemoryBacked : public virtual core::IReferenceCounted
{
    public:
        struct SDriverMemoryRequirements
        {
            VkMemoryRequirements vulkanReqs;

            uint32_t memoryHeapLocation             : 2; //IDeviceMemoryAllocation::E_SOURCE_MEMORY_TYPE
            uint32_t mappingCapability              : 4; //IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG
            uint32_t prefersDedicatedAllocation     : 1; /// Used and valid only in Vulkan
            uint32_t requiresDedicatedAllocation    : 1; /// Used and valid only in Vulkan
        };
        //! Combine requirements
        /** \return true on success, some requirements are mutually exclusive, so it may be impossible to combine them. */
        static inline bool combineRequirements(SDriverMemoryRequirements& out, const SDriverMemoryRequirements& a, const SDriverMemoryRequirements& b)
        {
            // TODO handle device local properly
            bool aDeviceLocal = (a.memoryHeapLocation & IDeviceMemoryAllocation::EMHF_DEVICE_LOCAL_BIT) != 0;
            bool bDeviceLocal = (b.memoryHeapLocation & IDeviceMemoryAllocation::EMHF_DEVICE_LOCAL_BIT) != 0;

            if (aDeviceLocal != bDeviceLocal)
                return false;
            out.memoryHeapLocation = aDeviceLocal;

            out.mappingCapability = a.mappingCapability|b.mappingCapability;
            out.prefersDedicatedAllocation = a.prefersDedicatedAllocation|b.prefersDedicatedAllocation;
            out.requiresDedicatedAllocation = a.requiresDedicatedAllocation|b.requiresDedicatedAllocation;

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
        virtual IDeviceMemoryAllocation* getBoundMemory() = 0;

        //! Constant version
        virtual const IDeviceMemoryAllocation* getBoundMemory() const = 0;

        //! Returns the offset in the allocation at which it is bound to the resource
        virtual size_t getBoundMemoryOffset() const = 0;

    protected:
        IDriverMemoryBacked() {}
        IDriverMemoryBacked(const SDriverMemoryRequirements& reqs) : cachedMemoryReqs(reqs) {}

        SDriverMemoryRequirements cachedMemoryReqs;
        // TODO: backward link to the IDeviceMemoryAllocation
        core::smart_refctd_ptr<IDeviceMemoryAllocation> m_backedMemory;
};

} // end namespace nbl::video

#endif
