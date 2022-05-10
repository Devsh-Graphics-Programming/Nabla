// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_BUFFER_H_INCLUDED__

#include "nbl/core/util/bitflag.h"

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
    public:
		struct SCachedCreationParams
		{
			size_t declaredSize = 0ull;
			core::bitflag<E_USAGE_FLAGS> usage = EUF_NONE;
			asset::E_SHARING_MODE sharingMode = asset::ESM_EXCLUSIVE;
			bool canUpdateSubRange = false; // whether `IGPUCommandBuffer::updateBuffer` can be used on this buffer
		};
		struct SCreationParams : SCachedCreationParams
		{
			uint32_t queueFamilyIndexCount = 0u;
			const uint32_t* queueFamilyIndices = nullptr;
		};
		
		E_OBJECT_TYPE getObjectType() const override { return EOT_BUFFER; }

		inline uint64_t getSize() const override {return m_cachedCreationParams.declaredSize;}

		inline const SCachedCreationParams& getCachedCreationParams() const {return m_cachedCreationParams;}

		// OpenGL: const GLuint* handle of a Buffer
		// Vulkan: const VkBuffer*
		virtual const void* getNativeHandle() const = 0;
		
    protected:
        IGPUBuffer(
			core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			const IDriverMemoryBacked::SDriverMemoryRequirements& reqs,
			const SCachedCreationParams& cachedCreationParams
		) : IDriverMemoryBacked(reqs), IBackendObject(std::move(dev)), m_cachedCreationParams(cachedCreationParams)
		{
		}

		const SCachedCreationParams m_cachedCreationParams;
};

} // end namespace nbl::video

#endif

