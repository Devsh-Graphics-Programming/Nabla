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
	friend COpenGLSwapchain;
	friend COpenGLESSwapchain;
	protected:
		virtual ~COpenGLImage();

		using GLuint = uint32_t;
		using GLenum = uint32_t;

		GLenum internalFormat;
		GLenum target;
		GLuint name;
	public:
		//! constructor
		COpenGLImage(
			core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			const uint32_t deviceLocalMemoryTypeBits,
			IGPUImage::SCreationParams&& _params,
			GLenum internalFormat,
			GLenum target,
			GLuint name
		) : IGPUImage(
				std::move(dev),
				SDeviceMemoryRequirements{0xdeadbeefBADC0FFEull,deviceLocalMemoryTypeBits,63u,true,true},
				std::move(_params)
			), IOpenGLMemoryAllocation(getOriginDevice()), internalFormat(internalFormat), target(target), name(name)
		{
			assert(name!=0u);
			m_cachedMemoryReqs.size = getImageDataSizeInBytes();
		}
		

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

