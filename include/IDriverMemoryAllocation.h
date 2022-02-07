// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_DRIVER_MEMORY_ALLOCATION_H_INCLUDED__
#define __NBL_I_DRIVER_MEMORY_ALLOCATION_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl
{
namespace video
{
//! Class corresponding to VkDeviceMemory and emulating them on OpenGL
/** This class replaces and takes over the functionality from the
old-alpha-version IGPUMappedBuffer class.
TO COPY BETWEEN MEMORY ALLOCATIONS you need to have them bound to
one or two IGPUBuffers and execute IVideoDriver::copyBuffer between them.
We only support persistently mapped buffers with ARB_buffer_storage.
Please don't ask us to support Buffer Orphaning. */
class IDriverMemoryAllocation : public virtual core::IReferenceCounted
{
public:
    //!
    struct MemoryRange
    {
        MemoryRange(const size_t& off, const size_t& len)
            : offset(off), length(len) {}

        size_t offset;
        size_t length;
    };

    //! Similar to VkMappedMemoryRange but no pNext
    struct MappedMemoryRange
    {
        MappedMemoryRange(IDriverMemoryAllocation* mem, const size_t& off, const size_t& len)
            : memory(mem), range(off, len) {}

        IDriverMemoryAllocation* memory;
        union
        {
            MemoryRange range;
            struct
            {
                size_t offset;
                size_t length;
            };
        };
    };

    //! Enumeration for Driver allocated memory location
    /**  For specifying your wish as to where you want the memory to live.
        This can only be guaranteed on Vulkan, in OpenGL these are just hints.
        ESMT_DONT_CARE is for OpenGL usage only, illegal in Vulkan. */
    enum E_SOURCE_MEMORY_TYPE
    {
        ESMT_DEVICE_LOCAL = 0u,
        ESMT_NOT_DEVICE_LOCAL,
        ESMT_DONT_KNOW,  ///< invalid in Vulkan
        ESMT_COUNT
    };
    //! Access flags for how the application plans to use mapped memory (if any)
    /** When you create the memory you can allow for it to be mapped (be given a pointer)
        for reading and writing directly from it, however the driver needs to know up-front
        about what you will do with it when allocating the memory so that it is allocated
        from the correct heap.
        If you don't match your creation and mapping flags then
        you will get errors and undefined behaviour. */
    enum E_MAPPING_CPU_ACCESS_FLAG
    {
        EMCAF_NO_MAPPING_ACCESS = 0x0u,
        EMCAF_READ = 0x1u,
        EMCAF_WRITE = 0x2u,
        EMCAF_READ_AND_WRITE = 0x3u
    };
    //! Memory mapping capability flags
    /** Depending on their creation flags (E_MAPPING_CPU_ACCESS_FLAG) memory allocations
        will have different capabilities in terms of mapping (direct memory transfer). */
    enum E_MAPPING_CAPABILITY_FLAGS
    {
        EMCF_CANNOT_MAP = EMCAF_NO_MAPPING_ACCESS,
        EMCF_CAN_MAP_FOR_READ = EMCAF_READ,
        EMCF_CAN_MAP_FOR_WRITE = EMCAF_WRITE,
        EMCF_COHERENT = 0x04u,  ///< whether mapping is coherent, i.e. no need to flush, which always true on read-enabled mappings.
        EMCF_CACHED = 0x08u,  ///< whether mapping is cached, i.e. if cpu reads go through cache, this is relevant to Vulkan only and is transparent to program operation.
    };

    //! Where the memory was actually allocated
    virtual E_SOURCE_MEMORY_TYPE getType() const { return ESMT_DONT_KNOW; }

    //! Utility function, tells whether the allocation can be mapped (whether mapMemory will ever return anything other than nullptr)
    inline bool isMappable() const { return this->getMappingCaps() != EMCF_CANNOT_MAP; }

    //! Utility function, tell us if writes by the CPU or GPU need extra visibility operations to become visible for reading on the other processor
    /** Only execute flushes or invalidations if the allocation requires them, and batch them (flush one combined range instead of two or more)
        for greater efficiency. To execute a flush or invalidation, use IDriver::flushMappedAllocationRanges and IDriver::invalidateMappedAllocationRanges respectively. */
    inline bool haveToMakeVisible() const
    {
        auto caps = this->getMappingCaps();
        return (caps & EMCF_COHERENT) == 0u;
    }

    //! Returns the size of the memory allocation
    virtual size_t getAllocationSize() const = 0;

    //! For details @see E_MAPPING_CAPABILITY_FLAGS
    virtual E_MAPPING_CAPABILITY_FLAGS getMappingCaps() const { return EMCF_CANNOT_MAP; }

    //! Utility function, tells if mapMemoryRange has been already called and unmapMemory was not called after.
    inline bool isCurrentlyMapped() const { return mappedPtr != nullptr; }

    //! Returns the currently mapped range, only valid if isCurrentlyMapped()==true
    inline MemoryRange getCurrentMappedRange() const { return mappedRange; }

    //!
    inline E_MAPPING_CPU_ACCESS_FLAG getCurrentMappingCaps() const { return currentMappingAccess; }

    //! Maps the memory sub-range of the allocation for reading, writing or both, @see getMappingCaps and @see getMappedPointer.
    /** This differs from the pointer returned by getMappedPointer, as it already has the offset of the mapping applied to the base pointer.
        Accessing the memory using the returned pointer with an offset which results in an address before or after the mapped range,
        or after calling unmapeMemory, will cause undefined behaviour, including program termination.
        For further advice and restrictions on the pointer usage @see getMappedPointer
		@returns Internal pointer to access driver allocated memory with the offset already applied. */
    virtual void* mapMemoryRange(const E_MAPPING_CPU_ACCESS_FLAG& accessType, const MemoryRange& memrange) { return nullptr; }

    //! Gets internal pointer.
    /** It is best you use a GPU Fence to ensure any operations that you have queued up which are or will be writing to this memory
        or reading from it have completed before you start using the returned pointer. Otherwise this will result in a race condition.
		WARNING: UNMAP will invalidate pointer!
        WARNING: NEED TO FENCE BEFORE USE!
		@returns Internal pointer with 0 offset into the memory allocation, so the address that it is pointing to may be unsafe
		to access without an offset if a memory range (if a subrange not starting at 0 was mapped). */
    inline void* getMappedPointer() { return mappedPtr; }

    //! Constant variant of getMappedPointer
    inline const void* getMappedPointer() const { return mappedPtr; }

    //! Unmaps mapped memory
    /** Unmaps memory, does not perform the implicit flush even in OpenGL (because we use ARB_buffer_storage).
        Any pointers obtained through mapMemoryRange or getMappedPointer before invoking this function will become invalid to use.
        */
    virtual void unmapMemory() = 0;

    //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
    virtual bool isDedicated() const = 0;

protected:
    IDriverMemoryAllocation()
        : mappedPtr(nullptr), mappedRange(0, 0), currentMappingAccess(EMCAF_NO_MAPPING_ACCESS) {}

    uint8_t* mappedPtr;
    MemoryRange mappedRange;
    E_MAPPING_CPU_ACCESS_FLAG currentMappingAccess;
};

}  // end namespace scene
}  // end namespace nbl

#endif
