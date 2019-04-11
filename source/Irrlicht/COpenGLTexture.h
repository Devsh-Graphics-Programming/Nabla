// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_TEXTURE_H_INCLUDED__
#define __C_OPEN_GL_TEXTURE_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "ITexture.h"
#include "IImage.h"
#include "COpenGLStateManager.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{

//! OpenGL texture.
class COpenGLTexture
{
public:
	//! return open gl texture name
	const GLuint& getOpenGLName() const {return TextureName;}
	GLuint* getOpenGLNamePtr() {return &TextureName;}

	//!
	const uint64_t& hasOpenGLNameChanged() const {return TextureNameHasChanged;}

	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const = 0;



    //!
    static bool isInternalFormatCompressed(GLenum format);

	//! Get the OpenGL color format parameters based on the given Irrlicht color format
	static void getOpenGLFormatAndParametersFromColorFormat(const asset::E_FORMAT &format, GLenum& colorformat, GLenum& type); //kill this

	static GLint getOpenGLFormatAndParametersFromColorFormat(const asset::E_FORMAT &format);

	//!
	static asset::E_FORMAT getColorFormatFromSizedOpenGLFormat(const GLenum& sizedFormat);

	//! Get the OpenGL color format parameters based on the given Irrlicht color format
	static uint32_t getOpenGLFormatBpp(const GLenum& colorformat);

protected:
	//! protected constructor with basic setup, no GL texture name created, for derived classes
	COpenGLTexture(const GLenum& textureType_Target);

	//! destructor
	virtual ~COpenGLTexture();

	//! for resizes
    void recreateName(const GLenum& textureType_Target);

	GLuint TextureName;
	uint64_t TextureNameHasChanged;
private:
    COpenGLTexture() {}
};

//! .
class COpenGLFilterableTexture : public ITexture, public COpenGLTexture, public IDriverMemoryAllocation
{
public:
    virtual IVirtualTexture::E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const {return IVirtualTexture::EVTT_OPAQUE_FILTERABLE;}

	//! Get size
	virtual const uint32_t* getSize() const {return TextureSize;}
    virtual core::dimension2du getRenderableSize() const {return *reinterpret_cast<const core::dimension2du*>(TextureSize);}

	//! returns driver type of texture (=the driver, that created it)
	virtual E_DRIVER_TYPE getDriverType() const {return EDT_OPENGL;}

	//! returns color format of texture
	virtual asset::E_FORMAT getColorFormat() const {return ColorFormat;}

	//! returns pitch of texture (in bytes)
	virtual uint32_t getPitch() const {return video::getBitsPerPixelFromFormat(ColorFormat)*TextureSize[0]/8;}

	//!
	GLint getOpenGLInternalFormat() const {return InternalFormat;}

	virtual uint32_t getMipMapLevelCount() const {return MipLevelsStored;}

	//! return whether this texture has mipmaps
	virtual bool hasMipMaps() const {return MipLevelsStored>1;}

	//! Regenerates the mip map levels of the texture.
	virtual void regenerateMipMapLevels();


    virtual size_t getAllocationSize() const {return (TextureSize[2]*TextureSize[1]*getPitch()*3u)/2u;} // MipLevelsStored rough estimate
    virtual IDriverMemoryAllocation* getBoundMemory() {return this;}
    virtual const IDriverMemoryAllocation* getBoundMemory() const {return this;}
    virtual size_t getBoundMemoryOffset() const {return 0ll;}

    virtual E_SOURCE_MEMORY_TYPE getType() const {return ESMT_DEVICE_LOCAL;}
    virtual void unmapMemory() {}
    virtual bool isDedicated() const {return true;}

protected:
	//! protected constructor with basic setup, no GL texture name created, for derived classes
	COpenGLFilterableTexture(const io::path& name, const GLenum& textureType_Target);


	uint32_t TextureSize[3];
	uint32_t MipLevelsStored;

	GLint InternalFormat;
	asset::E_FORMAT ColorFormat;
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_

