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

		COpenGLImageView(SCreationParams&& _params) : IGPUImageView(std::move(_params)), name(0u), target(GL_INVALID_ENUM), internalFormat(GL_INVALID_ENUM)
		{
			target = ViewTypeToGLenumTarget[params.viewType];
			COpenGLExtensionHandler::extGlCreateTextures(target, 1, &name);
			internalFormat = getSizedOpenGLFormatFromOurFormat(params.format);
			COpenGLExtensionHandler::extGlTextureView(name, target, static_cast<COpenGLImage*>(params.image.get())->getOpenGLName(), internalFormat, );
		}

		void regenerateMipMapLevels() override
		{
			if (params. <= 1u)
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