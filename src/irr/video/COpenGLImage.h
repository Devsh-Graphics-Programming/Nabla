#ifndef __C_OPEN_GL_IMAGE_H_INCLUDED__
#define __C_OPEN_GL_IMAGE_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/video/IGPUImage"

#ifdef _IRR_COMPILE_WITH_OPENGL_


namespace irr
{
namespace video
{

class COpenGLImage : public IGPUImage
{
	public:
		//! constructor
		COpenGLImage(GLenum internalFormat, const uint32_t* size, uint32_t mipmapLevels);

		//! Returns the allocation which is bound to the resource
		virtual IDriverMemoryAllocation* getBoundMemory() override;
		//! Constant version
		virtual const IDriverMemoryAllocation* getBoundMemory() const override;
		//! Returns the offset in the allocation at which it is bound to the resource
		virtual size_t getBoundMemoryOffset() const override;

		//! returns the opengl texture type
		virtual GLenum getOpenGLTextureType() const {return GL_TEXTURE_;}


		virtual bool updateSubRegion(const asset::E_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap=0, const uint32_t& unpackRowByteAlignment=0);
		virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0);
};


} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_OPENGL_

#endif

