// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__
#define __NBL_VIDEO_C_OPEN_GL_IMAGE_H_INCLUDED__

#include "BuildConfigOptions.h"

#include "nbl/video/IGPUImage.h"

#include "nbl/video/COpenGLCommon.h"
#include "nbl/video/IOpenGL_FunctionTable.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_


namespace nbl
{
namespace video
{

class IOpenGL_LogicalDevice;

class COpenGLImage final : public IGPUImage, public IDriverMemoryAllocation
{
	protected:
		virtual ~COpenGLImage();

		IOpenGL_LogicalDevice* m_device;
		GLenum internalFormat;
		GLenum target;
		GLuint name;

	public:
		//! constructor
		COpenGLImage(IOpenGL_LogicalDevice* dev, IOpenGL_FunctionTable* gl, IGPUImage::SCreationParams&& _params) : IGPUImage(std::move(_params)),
			m_device(dev), internalFormat(GL_INVALID_ENUM), target(GL_INVALID_ENUM), name(0u)
		{
			#ifdef OPENGL_LEAK_DEBUG
				COpenGLExtensionHandler::textureLeaker.registerObj(this);
			#endif // OPENGL_LEAK_DEBUG
			internalFormat = getSizedOpenGLFormatFromOurFormat(params.format);
			switch (params.type)
			{
				case IGPUImage::ET_1D:
					target = gl->TEXTURE_1D_ARRAY;
					gl->extGlCreateTextures(target, 1, &name);
					gl->extGlTextureStorage2D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width, params.arrayLayers);
					break;
				case IGPUImage::ET_2D:
					target = GL_TEXTURE_2D_ARRAY;
					gl->extGlCreateTextures(target, 1, &name);
					gl->extGlTextureStorage3D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width, params.extent.height, params.arrayLayers);
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


} // end namespace video
} // end namespace nbl

#endif // _NBL_COMPILE_WITH_OPENGL_

#endif

