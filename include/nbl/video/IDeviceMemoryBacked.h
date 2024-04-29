// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_DRIVER_MEMORY_BACKED_H_INCLUDED_
#define _NBL_VIDEO_I_DRIVER_MEMORY_BACKED_H_INCLUDED_


#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IDeviceMemoryAllocation.h"

#include <algorithm>


namespace nbl::video
{


//! If you bound an "exotic" memory object to the resource, you might require "special" cleanups in the destructor
struct NBL_API2 ICleanup
{
    virtual ~ICleanup() = 0;
};

//! Interface from which resources backed by IDeviceMemoryAllocation inherit from
class IDeviceMemoryBacked : public IBackendObject
{
    public:
        //!
        struct SCachedCreationParams
        {
            // A Pre-Destroy-Step is called out just before a `vkDestory` or `glDelete`, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> preDestroyCleanup = nullptr;
            // A Post-Destroy-Step is called in this class' destructor, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> postDestroyCleanup = nullptr;
            // If more than one, then we're doing concurrent resource sharing
            uint8_t queueFamilyIndexCount = 0u;
            // Thus the destructor will skip the call to `vkDestroy` or `glDelete` on the handle, this is only useful for "imported" objects
            bool skipHandleDestroy = false;

            //! If you specify multiple queue family indices, then you're concurrent sharing
            inline bool isConcurrentSharing() const
            {
                return queueFamilyIndexCount>1u;
            }
        };
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

        //!
		struct SMemoryBinding
		{
			inline bool isValid() const
			{
				return memory && offset<memory->getAllocationSize();
			}

			IDeviceMemoryAllocation* memory = nullptr;
			size_t offset = 0u;
		};
        //! Returns the allocation which is bound to the resource
        virtual SMemoryBinding getBoundMemory() const = 0;

        //! For constructor parameter only
        struct SCreationParams : SCachedCreationParams
        {
            const uint32_t* queueFamilyIndices = nullptr;
        };

    protected:
        inline IDeviceMemoryBacked(core::smart_refctd_ptr<const ILogicalDevice>&& originDevice, SCreationParams&& creationParams, const SDeviceMemoryRequirements& reqs)
            : IBackendObject(std::move(originDevice)), m_cachedCreationParams(std::move(creationParams)), m_cachedMemoryReqs(reqs) {}
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
};

} // end namespace nbl::video

#endif