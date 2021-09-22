// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_BUFFER_H_INCLUDED__


#include "nbl/asset/IBuffer.h"
#include "nbl/asset/IDescriptor.h"

#include "nbl/asset/ECommonEnums.h"
#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IDriverMemoryBacked.h"


namespace nbl::video
{

//! GPU Buffer class, where the memory is provided by the driver, does not support resizing.
/** For additional OpenGL DSA state-free operations such as flushing mapped ranges or
buffer to buffer copies, one needs a command buffer in Vulkan as these operations are
performed by the GPU and not wholly by the driver, so look for them in IGPUCommandBuffer. */
class IGPUBuffer : public asset::IBuffer, public IDriverMemoryBacked, public IBackendObject
{
    protected:
        IGPUBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IDriverMemoryBacked::SDriverMemoryRequirements& reqs) : IDriverMemoryBacked(reqs), IBackendObject(std::move(dev)) {}

    public:
		struct SCreationParams
		{
			core::bitflag<E_USAGE_FLAGS> usage = EUF_NONE;
			asset::E_SHARING_MODE sharingMode = asset::ESM_CONCURRENT;
			uint32_t queueFamilyIndexCount = 0u;
			const uint32_t* queueFamilyIndices = nullptr;
		};

        //! Get usable buffer byte size.
        inline const uint64_t& getSize() const {return cachedMemoryReqs.vulkanReqs.size;}

        //! Whether calling updateSubRange will produce any effects.
        virtual bool canUpdateSubRange() const = 0;
};

} // end namespace nbl::video

#endif

