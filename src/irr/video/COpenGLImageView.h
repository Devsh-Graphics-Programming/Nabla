#ifndef __IRR_C_OPENGL_IMAGE_VIEW_H_INCLUDED__
#define __IRR_C_OPENGL_IMAGE_VIEW_H_INCLUDED__


#include "irr/video/IGPUImageView.h"
#include "irr/video/COpenGLImage.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
namespace irr
{
namespace video
{

class COpenGLImageView final : public IGPUImageView
{
	protected:
		virtual ~COpenGLImageView()
		{
			if (name)
				glDeleteTextures(1u,&name);
		}

		GLuint name;
		GLenum target;
		GLenum internalFormat;

	public:
		_IRR_STATIC_INLINE_CONSTEXPR GLenum ViewTypeToGLenumTarget[IGPUImageView::ET_COUNT] = {
			GL_TEXTURE_1D,GL_TEXTURE_2D,GL_TEXTURE_3D,GL_TEXTURE_CUBE_MAP,GL_TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_CUBE_MAP_ARRAY
		};
		_IRR_STATIC_INLINE_CONSTEXPR GLenum ComponentMappingToGLenumSwizzle[IGPUImageView::SComponentMapping::ES_COUNT] = {GL_INVALID_ENUM,GL_ZERO,GL_ONE,GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA};

		COpenGLImageView(SCreationParams&& _params) : IGPUImageView(std::move(_params)), name(0u), target(GL_INVALID_ENUM), internalFormat(GL_INVALID_ENUM)
		{
			target = ViewTypeToGLenumTarget[params.viewType];
			COpenGLExtensionHandler::extGlCreateTextures(target, 1, &name);
			internalFormat = getSizedOpenGLFormatFromOurFormat(params.format);
			COpenGLExtensionHandler::extGlTextureView(	name, target, static_cast<COpenGLImage*>(params.image.get())->getOpenGLName(), internalFormat, 
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
			COpenGLExtensionHandler::extGlTextureParamteriv(name,GL_TEXTURE_SWIZZLE_RGBA,swizzle);
		}

		void regenerateMipMapLevels() override
		{
			if (params.subresourceRange.levelCount <= 1u)
				return;

			COpenGLExtensionHandler::extGlGenerateTextureMipmap(name,target);
		}

		inline GLuint getOpenGLName() const { return name; }
		inline GLenum getOpenGLTextureType() const {return target;}
		//inline GLenum getInternalFormat() const { return internalFormat; }
};

}
}
#endif

#endif