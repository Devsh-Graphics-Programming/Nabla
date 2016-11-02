#ifndef __C_OPEN_GL_3D_TEXTURE_H_INCLUDED__
#define __C_OPEN_GL_3D_TEXTURE_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLTexture.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{

//! OpenGL texture.
class COpenGL3DTexture : public COpenGLTexture
{
public:

	//! constructor
	COpenGL3DTexture(core::vector3d<u32> size, GLenum format, GLenum inDataFmt, GLenum inDataTpe, const io::path& name, const void* data, void* mipmapData=0, COpenGLDriver* driver=0, u32 mipmapLevels=0);


	virtual const E_DIMENSION_COUNT getDimensionality() const {return EDC_THREE;}

    virtual const E_TEXTURE_TYPE getTextureType() const {return ETT_3D;}

	//! returns pitch of texture (in bytes)
	virtual u32 getPitch() const;


    //!
    virtual core::dimension2du getRenderableSize() const {return *reinterpret_cast<const core::dimension2du*>(TextureSize);}


	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const;

    virtual bool updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, s32 mipmap=0);
    virtual bool resize(const uint32_t* size, u32 mipLevels=0);


protected:
};

//! OpenGL texture.
class COpenGL2DTextureArray : public COpenGLTexture
{
public:

//! needs my attention
	//! constructor
	COpenGL2DTextureArray(core::vector3d<u32> size, ECOLOR_FORMAT format, const io::path& name, void* mipmapData, COpenGLDriver* driver=0, u32 mipmapLevels=0);


	virtual const E_DIMENSION_COUNT getDimensionality() const {return EDC_THREE;}

    virtual const E_TEXTURE_TYPE getTextureType() const {return ETT_2D_ARRAY;}

	//! returns pitch of texture (in bytes)
	virtual u32 getPitch() const;


    //!
    virtual core::dimension2du getRenderableSize() const {return *reinterpret_cast<const core::dimension2du*>(TextureSize);}



	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const;


    virtual bool updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, s32 mipmap=0);
    virtual bool resize(const uint32_t* size, u32 mipLevels=0);


protected:
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_

