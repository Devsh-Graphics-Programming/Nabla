// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__


#include "BuildConfigOptions.h" // ?

#include "nbl/video/IGPUImage.h"

#include "nbl/video/COpenGLCommon.h"
#include "nbl/video/IOpenGL_FunctionTable.h"


namespace nbl::video
{

class COpenGLImage final : public IGPUImage, public IDriverMemoryAllocation
{
	protected:
		virtual ~COpenGLImage();

		GLenum internalFormat;
		GLenum target;
		GLuint name;
	public:
		//! constructor
		COpenGLImage(core::smart_refctd_ptr<const ILogicalDevice>&& dev, IOpenGL_FunctionTable* gl, IGPUImage::SCreationParams&& _params) : IGPUImage(std::move(dev), std::move(_params)),
			internalFormat(GL_INVALID_ENUM), target(GL_INVALID_ENUM), name(0u)
		{
			#ifdef OPENGL_LEAK_DEBUG
				COpenGLExtensionHandler::textureLeaker.registerObj(this);
			#endif // OPENGL_LEAK_DEBUG
			internalFormat = getSizedOpenGLFormatFromOurFormat(gl, params.format);
			GLsizei samples = params.samples;
			switch (params.type) // TODO what about multisample targets?
			{
				case IGPUImage::ET_1D:
					target = gl->TEXTURE_1D_ARRAY;
					gl->extGlCreateTextures(target, 1, &name);
					gl->extGlTextureStorage2D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width, params.arrayLayers);
					break;
				case IGPUImage::ET_2D:
					target = samples>1 ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY : GL_TEXTURE_2D_ARRAY;
					gl->extGlCreateTextures(target, 1, &name);
					if (samples == 1)
						gl->extGlTextureStorage3D(name, target, params.mipLevels, internalFormat, params.extent.width, params.extent.height, params.arrayLayers);
					else
						gl->extGlTextureStorage3DMultisample(name, target, samples, internalFormat, params.extent.width, params.extent.height, params.arrayLayers, GL_TRUE);
					break;
				case IGPUImage::ET_3D:
					target = GL_TEXTURE_3D;
					gl->extGlCreateTextures(target, 1, &name);
					gl->extGlTextureStorage3D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width, params.extent.height, params.extent.depth);
					break;
				default:
					assert(false);
					break;
			}
		}

		//!
		inline GLenum getOpenGLSizedFormat() const { return internalFormat; }

		//!
		inline GLenum getOpenGLTarget() const { return target; }

		//! returns the opengl texture handle
		inline GLuint getOpenGLName() const { return name; }


		inline size_t getAllocationSize() const override { return this->getImageDataSizeInBytes(); }
		inline IDriverMemoryAllocation* getBoundMemory() override { return this; }
		inline const IDriverMemoryAllocation* getBoundMemory() const override { return this; }
		inline size_t getBoundMemoryOffset() const override { return 0ull; }

		inline E_SOURCE_MEMORY_TYPE getType() const override { return ESMT_DEVICE_LOCAL; }
		inline bool isDedicated() const override { return true; }
};

} // end namespace nbl::video


#endif

