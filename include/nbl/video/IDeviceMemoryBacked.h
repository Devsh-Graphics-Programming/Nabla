// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED_
#define _NBL_I_DRIVER_MEMORY_BACKED_H_INCLUDED_

#include "IDeviceMemoryAllocation.h"
#include <algorithm>

namespace nbl::video
{
//! If you bound an "exotic" memory object to the resource, you might require "special" cleanups in the destructor
struct NBL_API2 ICleanup
{
    virtual ~ICleanup() = 0;
};

//! Interface from which resources backed by IDeviceMemoryAllocation inherit from
class IDeviceMemoryBacked : public virtual core::IReferenceCounted
{
    public:
        //! Flags for imported/exported allocation
        enum E_EXTERNAL_HANDLE_TYPE : uint32_t
        {
            EHT_NONE = 0,
            EHT_OPAQUE_FD = 0x00000001,
            EHT_OPAQUE_WIN32 = 0x00000002,
            EHT_OPAQUE_WIN32_KMT = 0x00000004,
            EHT_D3D11_TEXTURE = 0x00000008,
            EHT_D3D11_TEXTURE_KMT = 0x00000010,
            EHT_D3D12_HEAP = 0x00000020,
            EHT_D3D12_RESOURCE = 0x00000040,
            EHT_HOST_ALLOCATION_BIT = 0x00000080,
            EHT_HOST_MAPPED_FOREIGN_MEMORY = 0x00000100,
            EHT_DMA_BUF = 0x00000200,
            EHT_ANDROID_HARDWARE_BUFFER = 0x00000400,
            EHT_ZIRCON_VMO = 0x00000800,
            EHT_RDMA_ADDRESS = 0x00001000,
            EHT_LAST = EHT_RDMA_ADDRESS,

            EHT_WIN32_TYPES = EHT_OPAQUE_WIN32
                            | EHT_OPAQUE_WIN32_KMT
                            | EHT_D3D11_TEXTURE
                            | EHT_D3D11_TEXTURE_KMT
                            | EHT_D3D12_HEAP
                            | EHT_D3D12_RESOURCE,
        };

        static constexpr uint32_t HANDLE_TYPE_COUNT = 1u + std::countr_zero(static_cast<uint32_t>(EHT_LAST));

        /* ExternalMemoryProperties *//* provided by VK_KHR_external_memory_capabilities */
        struct SExternalMemoryProperties
        {
            uint32_t exportableTypes : IDeviceMemoryBacked::HANDLE_TYPE_COUNT = ~0u;
            uint32_t compatibleTypes : IDeviceMemoryBacked::HANDLE_TYPE_COUNT = ~0u;
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

        //!
        struct SCachedCreationParams
        {
            // A Pre-Destroy-Step is called out just before a `vkDestory` or `glDelete`, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> preDestroyCleanup = nullptr;
            // A Post-Destroy-Step is called in this class' destructor, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> postDestroyCleanup = nullptr;
            // If non zero, then we're doing concurrent resource sharing
            uint8_t queueFamilyIndexCount = 0u;
            // Thus the destructor will skip the call to `vkDestroy` or `glDelete` on the handle, this is only useful for "imported" objects
            bool skipHandleDestroy = false;
            // Handle Type for external resources
            core::bitflag<E_EXTERNAL_HANDLE_TYPE> externalHandleTypes = EHT_NONE;
            //! Imports the given handle  if externalHandle != nullptr && externalHandleType != EHT_NONE
            //! Creates exportable memory if externalHandle == nullptr && externalHandleType != EHT_NONE
            void* externalHandle = nullptr;
            //! If you specify queue family indices, then you're concurrent sharing
            inline bool isConcurrentSharing() const
            {
                return queueFamilyIndexCount!=0u;
            }
        };

        //!
        inline const SCachedCreationParams& getCachedCreationParams() const {return m_cachedCreationParams;}

        //! We need to know to cast to `IGPUBuffer` or `IGPUImage`
        enum E_OBJECT_TYPE : bool
        {
            EOT_BUFFER,
            EOT_IMAGE
        };
        
        //! Return type of memory backed object (image or buffer)
        virtual E_OBJECT_TYPE getObjectType() const = 0;
        
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
        };

        static_assert(sizeof(SDeviceMemoryRequirements)==16);

        //! Before allocating memory from the driver or trying to bind a range of an existing allocation
        inline const SDeviceMemoryRequirements& getMemoryReqs() const {return m_cachedMemoryReqs;}

        //! Returns the allocation which is bound to the resource
        virtual IDeviceMemoryAllocation* getBoundMemory() = 0;

        //! Constant version
        virtual const IDeviceMemoryAllocation* getBoundMemory() const = 0;

        //! Get handle of external memory, might be null
        virtual void* getExternalHandle() { return nullptr;  }

        //! Check whether if the resource exportable as the requested type
        virtual bool isExportableAs(E_EXTERNAL_HANDLE_TYPE type) const { return false;  }

        //! Returns the offset in the allocation at which it is bound to the resource
        virtual size_t getBoundMemoryOffset() const = 0;

        //! For constructor parameter only
        struct SCreationParams : SCachedCreationParams
        {
            const uint32_t* queueFamilyIndices = nullptr;
        };

    protected:
        inline IDeviceMemoryBacked(SCreationParams&& _creationParams, const SDeviceMemoryRequirements& reqs)
            : m_cachedCreationParams(std::move(_creationParams)), m_cachedMemoryReqs(reqs)
        {
        }
        inline virtual ~IDeviceMemoryBacked()
        {
            assert(!m_cachedCreationParams.preDestroyCleanup); // derived class should have already cleared this out
        }

        // it's the derived class' responsibility to call this in its destructor
        inline void preDestroyStep()
        {
            m_cachedCreationParams.preDestroyCleanup = nullptr;
        }

        //! members
        SCachedCreationParams m_cachedCreationParams;
        SDeviceMemoryRequirements m_cachedMemoryReqs;
        void* m_cachedExternalHandle = nullptr;
};

} // end namespace nbl::video

#endif
