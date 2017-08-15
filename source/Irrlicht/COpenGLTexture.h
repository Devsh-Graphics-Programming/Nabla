// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_TEXTURE_H_INCLUDED__
#define __C_OPEN_GL_TEXTURE_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "ITexture.h"
#include "IImage.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{

class COpenGLDriver;
//! OpenGL texture.
class COpenGLTexture : public ITexture
{
public:

	//! constructor
	COpenGLTexture(IImage* surface, const io::path& name, void* mipmapData=0, COpenGLDriver* driver=0, uint32_t mipmapLevels=0);

	COpenGLTexture(GLenum internalFormat, core::dimension2du size, const void* data, GLenum inDataFmt, GLenum inDataTpe, const io::path& name, void* mipmapData=0, COpenGLDriver* driver=0, uint32_t mipmapLevels=0);

	//! destructor
	virtual ~COpenGLTexture();

	//! Returns size of the texture.
	virtual const E_DIMENSION_COUNT getDimensionality() const {return EDC_TWO;}

    virtual const E_TEXTURE_TYPE getTextureType() const {return ETT_2D;}


	virtual const uint32_t* getSize() const {return TextureSize;}
    virtual core::dimension2du getRenderableSize() const {return *reinterpret_cast<const core::dimension2du*>(TextureSize);}

	//! returns driver type of texture (=the driver, that created it)
	virtual E_DRIVER_TYPE getDriverType() const {return EDT_OPENGL;}

	//! returns color format of texture
	virtual ECOLOR_FORMAT getColorFormat() const;

	//! returns pitch of texture (in bytes)
	virtual uint32_t getPitch() const;

	//! return open gl texture name
	const GLuint& getOpenGLName() const {return TextureName;}
	GLuint* getOpenGLNamePtr() {return &TextureName;}

	GLint getOpenGLInternalFormat() const {return InternalFormat;}
	///GLenum getOpenGLPixelFormat() const;
	///GLenum getOpenGLPixelType() const;

	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const {return GL_TEXTURE_2D;}

	//! return whether this texture has mipmaps
	virtual bool hasMipMaps() const;

	//! Regenerates the mip map levels of the texture.
	/** Useful after locking and modifying the texture
	\param mipmapData Pointer to raw mipmap data, including all necessary mip levels, in the same format as the main texture image. If not set the mipmaps are derived from the main image. */
	virtual void regenerateMipMapLevels();


    virtual bool updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap=0);
    virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0);


	const uint64_t& hasOpenGLNameChanged() const {return TextureNameHasChanged;}

    //!
    static bool isInternalFormatCompressed(GLenum format);

	//! Get the OpenGL color format parameters based on the given Irrlicht color format
	static GLint getOpenGLFormatAndParametersFromColorFormat(const ECOLOR_FORMAT &format, GLenum& colorformat, GLenum& type);

	static GLint getOpenGLFormatAndParametersFromColorFormat(const ECOLOR_FORMAT &format)
	{
	     GLenum colorformat;
	     GLenum type;
	     return getOpenGLFormatAndParametersFromColorFormat(format,colorformat,type);
	}
protected:

	//! protected constructor with basic setup, no GL texture name created, for derived classes
	COpenGLTexture(const io::path& name, COpenGLDriver* driver);

	//! get the desired color format based on texture creation flags and the input format.
	ECOLOR_FORMAT getBestColorFormat(ECOLOR_FORMAT format);


	//! Get the OpenGL color format parameters based on the given Irrlicht color format
	uint32_t getOpenGLFormatBpp(const GLenum& colorformat) const;


	//! get important numbers of the image and hw texture
	core::dimension2du getImageValues(IImage* image);

	uint32_t TextureSize[3];
	ECOLOR_FORMAT ColorFormat;
	COpenGLDriver* Driver;

	GLuint TextureName;
	uint64_t TextureNameHasChanged;
	GLint InternalFormat;
	///GLenum PixelFormat;
	///GLenum PixelType;

	uint32_t MipLevelsStored;
	bool HasMipMaps;
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_

