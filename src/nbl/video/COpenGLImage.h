// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__


#include "BuildConfigOptions.h" // ?

#include "nbl/video/IGPUImage.h"

#include "nbl/video/COpenGLCommon.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/IOpenGLMemoryAllocation.h"


namespace nbl::video
{

class COpenGLImage final : public IGPUImage, public IOpenGLMemoryAllocation
{
	protected:
		virtual ~COpenGLImage();

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
			core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags) override
		{
			if(!IOpenGLMemoryAllocation::initMemory(gl, allocateFlags, memoryPropertyFlags))
				return false;
			GLsizei samples = m_creationParams.samples;
			switch (m_creationParams.type) // TODO what about multisample targets?
			{
				case IGPUImage::ET_1D:
					gl->extGlTextureStorage2D(
						name, target, m_creationParams.mipLevels, internalFormat,
						m_creationParams.extent.width, m_creationParams.arrayLayers
					);
					break;
				case IGPUImage::ET_2D:
					if (samples == 1)
						gl->extGlTextureStorage3D(
							name, target, m_creationParams.mipLevels, internalFormat,
							m_creationParams.extent.width, m_creationParams.extent.height, m_creationParams.arrayLayers
						);
					else
						gl->extGlTextureStorage3DMultisample(
							name, target, samples, internalFormat,
							m_creationParams.extent.width, m_creationParams.extent.height, m_creationParams.arrayLayers, GL_TRUE
						);
					break;
				case IGPUImage::ET_3D:
					gl->extGlTextureStorage3D(
						name, target, m_creationParams.mipLevels, internalFormat,
						m_creationParams.extent.width, m_creationParams.extent.height, m_creationParams.extent.depth
					);
					break;
				default:
					assert(false);
					break;
			}
			return true;
		}

		void setObjectDebugName(const char* label) const override;

		//!
		inline GLenum getOpenGLSizedFormat() const { return internalFormat; }

		//!
		inline GLenum getOpenGLTarget() const { return target; }

		//! returns the opengl texture handle
		inline const void* getNativeHandle() const override {return &name;}
		inline GLuint getOpenGLName() const {return name;}


		inline size_t getAllocationSize() const override { return this->getImageDataSizeInBytes(); }
		inline IDeviceMemoryAllocation* getBoundMemory() override { return this; }
		inline const IDeviceMemoryAllocation* getBoundMemory() const override { return this; }
		inline size_t getBoundMemoryOffset() const override { return 0ull; }

		inline bool isDedicated() const override { return true; }
};

} // end namespace nbl::video


#endif

