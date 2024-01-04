// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_DRIVER_MEMORY_ALLOCATION_H_INCLUDED_
#define _NBL_VIDEO_I_DRIVER_MEMORY_ALLOCATION_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"

#include "nbl/video/EApiType.h"

namespace nbl::video
{

//fwd decl
class ILogicalDevice;

//! Class corresponding to VkDeviceMemory and emulating them on OpenGL
/** This class replaces and takes over the functionality from the
old-alpha-version IGPUMappedBuffer class.
TO COPY BETWEEN MEMORY ALLOCATIONS you need to have them bound to
one or two IGPUBuffers and execute IVideoDriver::copyBuffer between them.
We only support persistently mapped buffers with ARB_buffer_storage.
Please don't ask us to support Buffer Orphaning. */
class IDeviceMemoryAllocation : public virtual core::IReferenceCounted
{
    public:
        //! Access flags for how the application plans to use mapped memory (if any)
        /** When you create the memory you can allow for it to be mapped (be given a pointer)
        for reading and writing directly from it, however the driver needs to know up-front
        about what you will do with it when allocating the memory so that it is allocated
        from the correct heap.
        If you don't match your creation and mapping flags the you will get errors and undefined behaviour. */
        enum E_MAPPING_CPU_ACCESS_FLAGS : uint8_t
        {
            EMCAF_NO_MAPPING_ACCESS=0x0u,
            EMCAF_READ=0x1u,
            EMCAF_WRITE=0x2u,
            EMCAF_READ_AND_WRITE=(EMCAF_READ|EMCAF_WRITE)
        };
        //! Memory allocate flags
        enum E_MEMORY_ALLOCATE_FLAGS : uint8_t
        {
            EMAF_NONE = 0x00000000,
            // EMAF_DEVICE_MASK_BIT = 0x00000001, // We'll just deduce it in the future from it being provided
            EMAF_DEVICE_ADDRESS_BIT = 0x00000002,
            // EMAF_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT = 0x00000004, // See notes in VulkanSpec and IDeviceMemoryAllocator::SAllocateInfo
        };
        enum E_MEMORY_PROPERTY_FLAGS : uint16_t
        {
            EMPF_NONE               = 0,
            EMPF_DEVICE_LOCAL_BIT   = 0x00000001,
            EMPF_HOST_READABLE_BIT  = 0x00000002, 
            EMPF_HOST_WRITABLE_BIT  = 0x00000004, 
            EMPF_HOST_COHERENT_BIT  = 0x00000008,
            EMPF_HOST_CACHED_BIT    = 0x000000010,
            //EMPF_LAZILY_ALLOCATED_BIT = 0x00000020,
            //EMPF_PROTECTED_BIT = 0x00000040,
            //EMPF_DEVICE_COHERENT_BIT_AMD = 0x00000080,
            //EMPF_DEVICE_UNCACHED_BIT_AMD = 0x00000100,
            //EMPF_RDMA_CAPABLE_BIT_NV = 0x00000200,
        };
        //
        enum E_MEMORY_HEAP_FLAGS : uint32_t
        {
            EMHF_NONE               = 0,
            EMHF_DEVICE_LOCAL_BIT   = 0x00000001,
            EMHF_MULTI_INSTANCE_BIT = 0x00000002,
        };

        //
        const ILogicalDevice* getOriginDevice() const {return m_originDevice;}

        //!
        E_API_TYPE getAPIType() const;

        //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
        inline bool isDedicated() const {return m_dedicated;}

        //! Returns the size of the memory allocation
        inline size_t getAllocationSize() const {return m_allocationSize;}

        //!
        inline core::bitflag<E_MEMORY_ALLOCATE_FLAGS> getAllocateFlags() const { return m_allocateFlags; }

        //!
        inline core::bitflag<E_MEMORY_PROPERTY_FLAGS> getMemoryPropertyFlags() const { return m_memoryPropertyFlags; }

        //! Utility function, tells whether the allocation can be mapped (whether mapMemory will ever return anything other than nullptr)
        inline bool isMappable() const {return m_memoryPropertyFlags.hasFlags(EMPF_HOST_READABLE_BIT)||m_memoryPropertyFlags.hasFlags(EMPF_HOST_WRITABLE_BIT);}
        //! Utility function, tell us if writes by the CPU or GPU need extra visibility operations to become visible for reading on the other processor
        /** Only execute flushes or invalidations if the allocation requires them, and batch them (flush one combined range instead of two or more)
        for greater efficiency. To execute a flush or invalidation, use IDriver::flushMappedAllocationRanges and IDriver::invalidateMappedAllocationRanges respectively. */
        inline bool haveToMakeVisible() const
        {
            return !m_memoryPropertyFlags.hasFlags(EMPF_HOST_COHERENT_BIT);
        }

        //!
        struct MemoryRange
        {
            size_t offset = 0ull;
            size_t length = 0ull;
        };
        inline void* map(const MemoryRange& range, const core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> accessHint=IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE)
        {
            if (isCurrentlyMapped())
                return nullptr;
            if(accessHint.hasFlags(EMCAF_READ) && !m_memoryPropertyFlags.hasFlags(EMPF_HOST_READABLE_BIT))
                return nullptr;
            if(accessHint.hasFlags(EMCAF_WRITE) && !m_memoryPropertyFlags.hasFlags(EMPF_HOST_WRITABLE_BIT))
                return nullptr;
            m_mappedPtr = reinterpret_cast<uint8_t*>(map_impl(range,accessHint));
            if (m_mappedPtr)
                m_mappedPtr -= range.offset;
            m_mappedRange = m_mappedPtr ? range:MemoryRange{};
            m_currentMappingAccess = m_mappedPtr ? EMCAF_NO_MAPPING_ACCESS:accessHint;
            return m_mappedPtr;
        }
        // returns true on success, false on failure
        inline bool unmap()
        {
            if (!isCurrentlyMapped())
                return false;
            if (!unmap_impl())
                return false;
            m_mappedPtr = nullptr;
            m_mappedRange = {};
            m_currentMappingAccess = EMCAF_NO_MAPPING_ACCESS;
            return true;
        }

        //!
        inline bool isCurrentlyMapped() const { return m_mappedPtr; }
        //! Only valid if `isCurrentlyMapped` is true
        inline const MemoryRange& getMappedRange() const { return m_mappedRange; }
        //! returns current mapping access based on latest mapMemory's "accessHint", has no effect on Nabla's Vulkan Backend
        inline core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> getCurrentMappingAccess() const {return m_currentMappingAccess;}

        //! Gets internal pointer.
        /** It is best you use a GPU Fence to ensure any operations that you have queued up which are or will be writing to this memory
        or reading from it have completed before you start using the returned pointer. Otherwise this will result in a race condition.
        WARNING: UNMAP will invalidate pointer!
        WARNING: NEED TO FENCE BEFORE USE!
        @returns Internal pointer with 0 offset into the memory allocation, so the address that it is pointing to may be unsafe
        to access without an offset if a memory range (if a subrange not starting at 0 was mapped). */
        inline void* getMappedPointer() { return m_mappedPtr; }

        //! Constant variant of getMappedPointer
        inline const void* getMappedPointer() const { return m_mappedPtr; }

    protected:
        inline IDeviceMemoryAllocation(
            const ILogicalDevice* const originDevice, const size_t _size, const core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags, const core::bitflag<E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags, const bool dedicated
        ) : m_originDevice(originDevice), m_allocationSize(_size), m_allocateFlags(allocateFlags), m_memoryPropertyFlags(memoryPropertyFlags), m_dedicated(dedicated) {}

        virtual void* map_impl(const MemoryRange& range, const core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> accessHint) = 0;
        virtual bool unmap_impl() = 0;


        const ILogicalDevice* const m_originDevice;
        const size_t m_allocationSize;
        uint8_t* m_mappedPtr = nullptr;
        MemoryRange m_mappedRange = {};
        core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> m_currentMappingAccess = EMCAF_NO_MAPPING_ACCESS;
        const core::bitflag<E_MEMORY_ALLOCATE_FLAGS> m_allocateFlags;
        const core::bitflag<E_MEMORY_PROPERTY_FLAGS> m_memoryPropertyFlags;
        const bool m_dedicated;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS)

} // end namespace nbl::video

#endif


