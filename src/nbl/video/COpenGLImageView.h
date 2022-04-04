// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_VIDEO_C_OPENGL_IMAGE_VIEW_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_IMAGE_VIEW_H_INCLUDED__


#include "nbl/video/IGPUImageView.h"
#include "nbl/video/COpenGLImage.h"


#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl::video
{

class COpenGLImageView final : public IGPUImageView
{
	protected:
		virtual ~COpenGLImageView();

		GLuint name;
		GLenum target;
		GLenum internalFormat;
	public:
		static inline constexpr GLenum ViewTypeToGLenumTarget[IGPUImageView::ET_COUNT] = {
			GL_TEXTURE_1D,GL_TEXTURE_2D,GL_TEXTURE_3D,GL_TEXTURE_CUBE_MAP,GL_TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_CUBE_MAP_ARRAY
		};
		static inline constexpr GLenum ComponentMappingToGLenumSwizzle[IGPUImageView::SComponentMapping::ES_COUNT] = {GL_INVALID_ENUM,GL_ZERO,GL_ONE,GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA};

		void setObjectDebugName(const char* label) const override;

		GLenum getOpenGLTarget() const
		{
			return target;
		}

		COpenGLImageView(core::smart_refctd_ptr<const ILogicalDevice>&& dev, IOpenGL_FunctionTable* gl, SCreationParams&& _params) :
			IGPUImageView(std::move(dev), std::move(_params)), name(0u), target(GL_INVALID_ENUM), internalFormat(GL_INVALID_ENUM)
		{
			target = ViewTypeToGLenumTarget[params.viewType];
			if (params.image->getCreationParameters().samples>1u)
			switch (target)
			{
				case GL_TEXTURE_2D:
					target = GL_TEXTURE_2D_MULTISAMPLE;
					break;
				case GL_TEXTURE_2D_ARRAY:
					target = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
					break;
				default:
					target = GL_INVALID_ENUM;
					break;
			}
			internalFormat = getSizedOpenGLFormatFromOurFormat(gl, params.format);
            assert(internalFormat != GL_INVALID_ENUM);

			//glTextureView spec:
			//GL_INVALID_OPERATION is generated if texture has already been bound or otherwise given a target.
			//thus we cannot create a name for view with glCreateTextures
			gl->glTexture.pglGenTextures(1, &name);
			gl->extGlTextureView(	name, target, IBackendObject::compatibility_cast<COpenGLImage*>(params.image.get(), this)->getOpenGLName(), internalFormat, 
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
		
		inline const void* getNativeHandle() const override {return &name;}
		inline GLuint getOpenGLName() const { return name; }
		inline GLenum getInternalFormat() const { return internalFormat; }
};

}
#endif

#endif