// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_IMAGE_VIEW_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_IMAGE_VIEW_H_INCLUDED__


#include "nbl/video/IGPUImageView.h"
#include "nbl/video/COpenGLImage.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
{
namespace video
{

class COpenGLImageView final : public IGPUImageView
{
	protected:
		virtual ~COpenGLImageView();

		GLuint name;
		GLenum target;
		GLenum internalFormat;
		core::smart_refctd_ptr<system::ILogger> m_logger;
	public:
		_NBL_STATIC_INLINE_CONSTEXPR GLenum ViewTypeToGLenumTarget[IGPUImageView::ET_COUNT] = {
			IOpenGL_FunctionTable::TEXTURE_1D,GL_TEXTURE_2D,GL_TEXTURE_3D,GL_TEXTURE_CUBE_MAP,IOpenGL_FunctionTable::TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_CUBE_MAP_ARRAY
		};
		_NBL_STATIC_INLINE_CONSTEXPR GLenum ComponentMappingToGLenumSwizzle[IGPUImageView::SComponentMapping::ES_COUNT] = {GL_INVALID_ENUM,GL_ZERO,GL_ONE,GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA};

		GLenum getOpenGLTarget() const
		{
			auto viewtype = params.viewType;
			GLenum target = ViewTypeToGLenumTarget[viewtype];
			return target;
		}

		COpenGLImageView(ILogicalDevice* dev, IOpenGL_FunctionTable* gl, SCreationParams&& _params, core::smart_refctd_ptr<system::ILogger>&& logger) :
			IGPUImageView(dev, std::move(_params)), name(0u), target(GL_INVALID_ENUM), internalFormat(GL_INVALID_ENUM), m_logger(std::move(logger))
		{
			target = ViewTypeToGLenumTarget[params.viewType];
			internalFormat = getSizedOpenGLFormatFromOurFormat(gl, params.format, m_logger.get());
            assert(internalFormat != GL_INVALID_ENUM);

			//glTextureView spec:
			//GL_INVALID_OPERATION is generated if texture has already been bound or otherwise given a target.
			//thus we cannot create a name for view with glCreateTextures
			gl->glTexture.pglGenTextures(1, &name);
			gl->extGlTextureView(	name, target, static_cast<COpenGLImage*>(params.image.get())->getOpenGLName(), internalFormat, 
														params.subresourceRange.baseMipLevel, params.subresourceRange.levelCount,
														params.subresourceRange.baseArrayLayer, params.subresourceRange.layerCount);

			GLint swizzle[4u] = {GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA};
			for (auto i=0u; i<4u; i++)
			{
				auto currentMapping = (&params.components.r)[i];
				if (currentMapping==IGPUImageView::SComponentMapping::ES_IDENTITY)
					continue;
				swizzle[i] = ComponentMappingToGLenumSwizzle[currentMapping];
			}
			constexpr GLenum pname[4] = { GL_TEXTURE_SWIZZLE_R, GL_TEXTURE_SWIZZLE_G, GL_TEXTURE_SWIZZLE_B, GL_TEXTURE_SWIZZLE_A };
			for (uint32_t i = 0u; i < 4u; ++i)
			{
				gl->extGlTextureParameteriv(name, target, pname[i], swizzle+i);
			}
		}

		inline GLuint getOpenGLName() const { return name; }
		inline GLenum getOpenGLTextureType() const {return target;}
		inline GLenum getInternalFormat() const { return internalFormat; }
};

}
}
#endif

#endif