// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_DRIVER_MEMORY_ALLOCATION_H_INCLUDED__
#define __NBL_I_DRIVER_MEMORY_ALLOCATION_H_INCLUDED__

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
        friend class IDeviceMemoryAllocator;
        friend class ILogicalDevice;

    public:
        //!
        struct MemoryRange
        {
            MemoryRange(const size_t& off, const size_t& len) : offset(off), length(len) {}

            size_t offset;
            size_t length;
        };

        //! Similar to VkMappedMemoryRange but no pNext
        struct MappedMemoryRange
        {
            MappedMemoryRange() : memory(nullptr), range(0u,0u) {}
            MappedMemoryRange(IDeviceMemoryAllocation* mem, const size_t& off, const size_t& len) : memory(mem), range(off,len) {}

            IDeviceMemoryAllocation* memory;
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

        //! Access flags for how the application plans to use mapped memory (if any)
        /** When you create the memory you can allow for it to be mapped (be given a pointer)
        for reading and writing directly from it, however the driver needs to know up-front
        about what you will do with it when allocating the memory so that it is allocated
        from the correct heap.
        If you don't match your creation and mapping flags then
        you will get errors and undefined behaviour. */
        enum E_MAPPING_CPU_ACCESS_FLAGS
        {
            EMCAF_NO_MAPPING_ACCESS=0x0u,
            EMCAF_READ=0x1u,
            EMCAF_WRITE=0x2u,
            EMCAF_READ_AND_WRITE=(EMCAF_READ|EMCAF_WRITE)
        };

        //! Memory allocate flags
        enum E_MEMORY_ALLOCATE_FLAGS : uint32_t
        {
            EMAF_NONE = 0x00000000,
            EMAF_DEVICE_MASK_BIT = 0x00000001,
            EMAF_DEVICE_ADDRESS_BIT = 0x00000002,
            // EMAF_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT = 0x00000004, // See notes in VulkanSpec and IDeviceMemoryAllocator::SAllocateInfo
        };
        
        enum E_MEMORY_PROPERTY_FLAGS : uint32_t
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
        
        enum E_MEMORY_HEAP_FLAGS : uint32_t
        {
            EMHF_NONE               = 0,
            EMHF_DEVICE_LOCAL_BIT   = 0x00000001,
            EMHF_MULTI_INSTANCE_BIT = 0x00000002,
        };

        //! Flags for imported/exported allocation
        enum E_EXTERNAL_HANDLE_TYPE : uint32_t
        {
            EHT_NONE = 0,
            EHT_OPAQUE_WIN32 = 0x00000002,
            EHT_OPAQUE_WIN32_KMT = 0x00000004,
            EHT_D3D11_TEXTURE = 0x00000008,
            EHT_D3D11_TEXTURE_KMT = 0x00000010,
            EHT_D3D12_HEAP = 0x00000020,
            EHT_D3D12_RESOURCE = 0x00000040,
        };

        /* ExternalMemoryProperties *//* provided by VK_KHR_external_memory_capabilities */
        struct SExternalMemoryProperties
        {
            uint32_t exportableTypes : 7 = ~0u;
            uint32_t compatibleTypes : 7 = ~0u;
            uint32_t dedicatedOnly : 1 = 0u;
            uint32_t exportable : 1 = ~0u;
            uint32_t importable : 1 = ~0u;

            bool operator == (SExternalMemoryProperties const& rhs) const = default;

            SExternalMemoryProperties operator &(SExternalMemoryProperties rhs) const
            {
                rhs.exportableTypes &= exportableTypes;
                rhs.compatibleTypes &= compatibleTypes;
                rhs.dedicatedOnly |= dedicatedOnly;
                rhs.exportable &= exportable;
                rhs.importable &= importable;
                return rhs;
            }
        };

        static_assert(sizeof(SExternalMemoryProperties) == sizeof(uint32_t));

        E_API_TYPE getAPIType() const;

        //! Utility function, tells whether the allocation can be mapped (whether mapMemory will ever return anything other than nullptr)
        inline bool isMappable() const {return params.memoryPropertyFlags.hasFlags(EMPF_HOST_READABLE_BIT) || params.memoryPropertyFlags.hasFlags(EMPF_HOST_WRITABLE_BIT);}

        //! Utility function, tell us if writes by the CPU or GPU need extra visibility operations to become visible for reading on the other processor
        /** Only execute flushes or invalidations if the allocation requires them, and batch them (flush one combined range instead of two or more)
        for greater efficiency. To execute a flush or invalidation, use IDriver::flushMappedAllocationRanges and IDriver::invalidateMappedAllocationRanges respectively. */
        inline bool haveToMakeVisible() const
        {
            return (!params.memoryPropertyFlags.hasFlags(EMPF_HOST_COHERENT_BIT));
        }

        //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
        virtual bool isDedicated() const = 0;

        //! Returns the size of the memory allocation
        virtual size_t getAllocationSize() const = 0;

        //! Returns the API handle of the memory allocation
        virtual const void* getNativeHandle() const = 0;

        //! returns current mapping access based on latest mapMemory's "accessHint", has no effect on Nabla's Vulkan Backend
        inline core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> getCurrentMappingAccess() const {return currentMappingAccess;}
        //!
        inline core::bitflag<E_MEMORY_ALLOCATE_FLAGS> getAllocateFlags() const {return params.allocateFlags;}
        //!
        inline core::bitflag<E_MEMORY_PROPERTY_FLAGS> getMemoryPropertyFlags() const {return params.memoryPropertyFlags; }
        inline E_EXTERNAL_HANDLE_TYPE getExternalHandleType() const {return params.externalHandleType; }

        inline bool isCurrentlyMapped() const { return mappedPtr != nullptr; }

        //! Only valid if `isCurrentlyMapped` is true
        inline const MemoryRange& getMappedRange() const { return mappedRange; }

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

        static inline bool isMappingAccessConsistentWithMemoryType(core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> access, core::bitflag<E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags)
        {
            if(access.hasFlags(EMCAF_READ))
                if(!memoryPropertyFlags.hasFlags(EMPF_HOST_READABLE_BIT))
                    return false;
            if(access.hasFlags(EMCAF_WRITE))
                if(!memoryPropertyFlags.hasFlags(EMPF_HOST_WRITABLE_BIT))
                    return false;
            return true;
        }

#if 0
        void* getExternalHandle() override
        {
            if (m_cachedExternalHandle)
                return m_cachedExternalHandle;

            auto& ccp = getCachedCreationParams();

            if (ccp.externalHandleTypes.value)
            {
                if (ccp.externalHandle)
                    return m_cachedExternalHandle = ccp.externalHandle;

                return m_cachedExternalHandle = getOriginDevice()->getExternalHandle(this);
            }

            return nullptr;
        }

        bool isExportableAs(E_EXTERNAL_HANDLE_TYPE type) const override
        {
            auto props = getOriginDevice()->getPhysicalDevice()->getExternalMemoryProperties(getCreationParams().usage, type);
            if (!props.exportable || !(props.exportableTypes & type))
                return false;
            return true;
        }
#endif

        struct SCreationParams
        {
            core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags = E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE;
            core::bitflag<E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags = E_MEMORY_PROPERTY_FLAGS::EMPF_NONE;
            E_EXTERNAL_HANDLE_TYPE externalHandleType = E_EXTERNAL_HANDLE_TYPE::EHT_NONE;
            void* externalHandle = nullptr;
        };

    protected:
        inline void postMapSetMembers(void* ptr, MemoryRange rng, core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> access)
        {
            mappedPtr = reinterpret_cast<uint8_t*>(ptr);
            mappedRange = rng;
            currentMappingAccess = access;
        }

        inline void setPostDestroyCleanup(std::unique_ptr<struct ICleanup>&& cleanup)
        {
            postDestroyCleanup = std::move(cleanup);
        }

        IDeviceMemoryAllocation(
            const ILogicalDevice* originDevice, SCreationParams&& params = {})
            : m_originDevice(originDevice)
            , mappedPtr(nullptr)
            , mappedRange(0,0)
            , currentMappingAccess(EMCAF_NO_MAPPING_ACCESS)
            , params(std::move(params))
        {}

        const ILogicalDevice* m_originDevice = nullptr;
        uint8_t* mappedPtr;
        MemoryRange mappedRange;
        core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS>    currentMappingAccess;
        SCreationParams params;
        std::unique_ptr<struct ICleanup> postDestroyCleanup = nullptr;
};

} // end namespace nbl::video

#endif


