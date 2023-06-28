// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_BUFFER_H_INCLUDED_

#include "nbl/core/util/bitflag.h"

#include "nbl/asset/IBuffer.h"

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IDeviceMemoryBacked.h"


namespace nbl::video
{

//! GPU Buffer class, where the memory is provided by the driver, does not support resizing.
/** For additional OpenGL DSA state-free operations such as flushing mapped ranges or
buffer to buffer copies, one needs a command buffer in Vulkan as these operations are
performed by the GPU and not wholly by the driver, so look for them in IGPUCommandBuffer. */
class IGPUBuffer : public asset::IBuffer, public IDeviceMemoryBacked
{
	public:
		struct SCreationParams : asset::IBuffer::SCreationParams, IDeviceMemoryBacked::SCreationParams
		{
			SCreationParams& operator =(const asset::IBuffer::SCreationParams& rhs)
			{
				static_cast<asset::IBuffer::SCreationParams&>(*this) = rhs;
				return *this;
			}
		};

		//
		inline E_OBJECT_TYPE getObjectType() const override { return EOT_BUFFER; }

		//
		inline uint64_t getDeviceAddress() const {return m_deviceAddress;}

		// Vulkan: const VkBuffer*
		virtual const void* getNativeHandle() const = 0;

	protected:
		inline IGPUBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _creationParams, const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs)
			: asset::IBuffer(_creationParams), IDeviceMemoryBacked(std::move(dev),std::move(_creationParams),reqs) {}

		uint64_t m_deviceAddress = 0ull;
};

} // end namespace nbl::video

#endif

