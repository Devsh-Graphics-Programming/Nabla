#ifndef __IRR_C_OPENGL_IMAGE_VIEW_H_INCLUDED__
#define __IRR_C_OPENGL_IMAGE_VIEW_H_INCLUDED__


#include "irr/video/IGPUImageView.h"

#include "COpenGLExtensionHandler.h"

namespace irr
{
namespace video
{

class COpenGLImageView final : public IGPUImageView
{
	protected:
		~COpenGLImageView();

	public:
		COpenGLImageView();

		void regenerateMipMapLevels() override;

		GLuint getOpenGLName() const;
		GLenum getInternalFormat() const;
};

}
}

#endif