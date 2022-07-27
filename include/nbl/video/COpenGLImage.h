// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__


#include "BuildConfigOptions.h" // ?

#include "nbl/video/IGPUImage.h"
#include "nbl/video/IOpenGLMemoryAllocation.h"


namespace nbl::video
{

class IOpenGL_FunctionTable;

class COpenGLImage final : public IGPUImage, public IOpenGLMemoryAllocation
{
	protected:
		virtual ~COpenGLImage();

		uint32_t internalFormat;
		uint32_t target;
		uint32_t name;
	public:
		//! constructor
		COpenGLImage(
			core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			const uint32_t deviceLocalMemoryTypeBits,
			IGPUImage::SCreationParams&& _params,
			uint32_t internalFormat,
			uint32_t target,
			uint32_t name
		) : IGPUImage(std::move(dev), SDeviceMemoryRequirements{ 0ull/*TODO-SIZE*/, deviceLocalMemoryTypeBits, 8u /*alignment=log2(256u)*/, true, true }, std::move(_params)),
			IOpenGLMemoryAllocation(getOriginDevice()), internalFormat(internalFormat), target(target), name(name)
		{}

		//! foreign constructor
		COpenGLImage(
			core::smart_refctd_ptr<const ILogicalDevice> && dev,
			IGPUImage::SCreationParams && _params,
			uint32_t internalFormat,
			uint32_t target,
			uint32_t name,
			core::smart_refctd_ptr<ISwapchain> _backingSwapchain = nullptr,
			uint32_t _backingSwapchainIx = 0
		) : IGPUImage(std::move(dev), std::move(_params), _backingSwapchain, _backingSwapchainIx),
			IOpenGLMemoryAllocation(getOriginDevice()), internalFormat(internalFormat), target(target), name(name)
		{}

		bool initMemory(
			IOpenGL_FunctionTable* gl,
			core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
			core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags) override;

		void setObjectDebugName(const char* label) const override;

		//!
		inline uint32_t getOpenGLSizedFormat() const { return internalFormat; }

		//!
		inline uint32_t getOpenGLTarget() const { return target; }

		//! returns the opengl texture handle
		inline const void* getNativeHandle() const override {return &name;}
		inline uint32_t getOpenGLName() const {return name;}


		inline size_t getAllocationSize() const override { return this->getImageDataSizeInBytes(); }
		inline IDeviceMemoryAllocation* getBoundMemory() override { return this; }
		inline const IDeviceMemoryAllocation* getBoundMemory() const override { return this; }
		inline size_t getBoundMemoryOffset() const override { return 0ull; }

		inline bool isDedicated() const override { return true; }
};

} // end namespace nbl::video


#endif

