// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_TEXTURE_H_INCLUDED__
#define __I_TEXTURE_H_INCLUDED__

#include "IFrameBuffer.h"
#include "IImage.h"
#include "dimension2d.h"
#include "EDriverTypes.h"
#include "path.h"

namespace irr
{
namespace video
{

enum E_TEXURE_BUFFER_OBJECT_FORMAT
{
    ///1
    ETBOF_R8=0,
    ETBOF_R16,
    ETBOF_R16F,
    ETBOF_R32F,
    ETBOF_R8I,
    ETBOF_R16I,
    ETBOF_R32I,
    ETBOF_R8UI,
    ETBOF_R16UI,
    ETBOF_R32UI,
    ///2
    ETBOF_RG8,
    ETBOF_RG16,
    ETBOF_RG16F,
    ETBOF_RG32F,
    ETBOF_RG8I,
    ETBOF_RG16I,
    ETBOF_RG32I,
    ETBOF_RG8UI,
    ETBOF_RG16UI,
    ETBOF_RG32UI,
    ///3
    ETBOF_RGB32F,
    ETBOF_RGB32I,
    ETBOF_RGB32UI,
    ///4
    ETBOF_RGBA8,
    ETBOF_RGBA16,
    ETBOF_RGBA16F,
    ETBOF_RGBA32F,
    ETBOF_RGBA8I,
    ETBOF_RGBA16I,
    ETBOF_RGBA32I,
    ETBOF_RGBA8UI,
    ETBOF_RGBA16UI,
    ETBOF_RGBA32UI,
    ETBOF_COUNT
};


//! Enumeration flags telling the video driver in which format textures should be created.
enum E_TEXTURE_CREATION_FLAG
{
	/** Forces the driver to create 16 bit textures always, independent of
	which format the file on disk has. When choosing this you may lose
	some color detail, but gain much speed and memory. 16 bit textures can
	be transferred twice as fast as 32 bit textures and only use half of
	the space in memory.
	When using this flag, it does not make sense to use the flags
	ETCF_ALWAYS_32_BIT, ETCF_OPTIMIZED_FOR_QUALITY, or
	ETCF_OPTIMIZED_FOR_SPEED at the same time. */
	ETCF_ALWAYS_16_BIT = 0x00000001,

	/** Forces the driver to create 32 bit textures always, independent of
	which format the file on disk has. Please note that some drivers (like
	the software device) will ignore this, because they are only able to
	create and use 16 bit textures.
	When using this flag, it does not make sense to use the flags
	ETCF_ALWAYS_16_BIT, ETCF_OPTIMIZED_FOR_QUALITY, or
	ETCF_OPTIMIZED_FOR_SPEED at the same time. */
	ETCF_ALWAYS_32_BIT = 0x00000002,

	/** Lets the driver decide in which format the textures are created and
	tries to make the textures look as good as possible. Usually it simply
	chooses the format in which the texture was stored on disk.
	When using this flag, it does not make sense to use the flags
	ETCF_ALWAYS_16_BIT, ETCF_ALWAYS_32_BIT, or ETCF_OPTIMIZED_FOR_SPEED at
	the same time. */
	ETCF_OPTIMIZED_FOR_QUALITY = 0x00000004,

	/** Lets the driver decide in which format the textures are created and
	tries to create them maximizing render speed.
	When using this flag, it does not make sense to use the flags
	ETCF_ALWAYS_16_BIT, ETCF_ALWAYS_32_BIT, or ETCF_OPTIMIZED_FOR_QUALITY,
	at the same time. */
	ETCF_OPTIMIZED_FOR_SPEED = 0x00000008,

	/** Automatically creates mip map levels for the textures. */
	ETCF_CREATE_MIP_MAPS = 0x00000010,

	/** Discard any alpha layer and use non-alpha color format. */
	ETCF_NO_ALPHA_CHANNEL = 0x00000020,

	//! Allow the Driver to use Non-Power-2-Textures
	/** BurningVideo can handle Non-Power-2 Textures in 2D (GUI), but not in 3D. */
	ETCF_ALLOW_NON_POWER_2 = 0x00000040,

	/** This flag is never used, it only forces the compiler to compile
	these enumeration values to 32 bit. */
	ETCF_FORCE_32_BIT_DO_NOT_USE = 0x7fffffff
};


//! Interface of a Video Driver dependent Texture.
/** An ITexture is created by an IVideoDriver by using IVideoDriver::addTexture
or IVideoDriver::getTexture. After that, the texture may only be used by this
VideoDriver. As you can imagine, textures of the DirectX and the OpenGL device
will, e.g., not be compatible. An exception is the Software device and the
NULL device, their textures are compatible. If you try to use a texture
created by one device with an other device, the device will refuse to do that
and write a warning or an error message to the output buffer.
*/
class ITexture : public IRenderable
{
public:
    enum E_DIMENSION_COUNT
    {
        EDC_ZERO=0,
        EDC_ONE,
        EDC_TWO,
        EDC_THREE,
        EDC_COUNT,
        EDC_FORCE32BIT=0xffffffffu
    };
    enum E_TEXTURE_TYPE
    {
        ETT_1D=0,
        ETT_2D,
        ETT_3D,
        ETT_1D_ARRAY,
        ETT_2D_ARRAY,
        ETT_CUBE_MAP,
        ETT_CUBE_MAP_ARRAY,
        ETT_TEXTURE_BUFFER,
        ETT_COUNT
    };

	//! constructor
	ITexture(const io::path& name) : NamedPath(name)
	{
	}

	virtual const E_DIMENSION_COUNT getDimensionality() const = 0;

	//! Get dimension (=size) of the texture.
	/** \return The size of the texture. */
	virtual const uint32_t* getSize() const = 0;


	//!
    virtual bool updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, s32 mipmap=0) = 0;
    virtual bool resize(const uint32_t* size, const u32& mipLevels=0) = 0;

    //!
    virtual const E_TEXTURE_TYPE getTextureType() const = 0;

	//! Get driver type of texture.
	/** This is the driver, which created the texture. This method is used
	internally by the video devices, to check, if they may use a texture
	because textures may be incompatible between different devices.
	\return Driver type of texture. */
	virtual E_DRIVER_TYPE getDriverType() const = 0;

	//! Get the color format of texture.
	/** \return The color format of texture. */
	virtual ECOLOR_FORMAT getColorFormat() const = 0;

	//! Get pitch of the main texture (in bytes).
	/** The pitch is the amount of bytes used for a row of pixels in a
	texture.
	\return Pitch of texture in bytes. */
	virtual u32 getPitch() const = 0;

	//! Check whether the texture has MipMaps
	/** \return True if texture has MipMaps, else false. */
	virtual bool hasMipMaps() const { return false; }

	//virtual bool is3D() const { return false; }

	//! Returns if the texture has an alpha channel
	virtual bool hasAlpha() const {
		return getColorFormat () == video::ECF_A8R8G8B8 || video::ECF_R8G8B8A8 || getColorFormat () == video::ECF_A1R5G5B5 || getColorFormat () == video::ECF_A16B16G16R16F || getColorFormat () == ECF_A32B32G32R32F
                                            || getColorFormat() == ECF_RGBA_BC1 || getColorFormat() == ECF_RGBA_BC2 || getColorFormat() == ECF_RGBA_BC3;
	}

	//! Regenerates the mip map levels of the texture.
	/** Required after modifying the texture, usually after calling unlock().
	\param mipmapData Optional parameter to pass in image data which will be
	used instead of the previously stored or automatically generated mipmap
	data. The data has to be a continuous pixel data for all mipmaps until
	1x1 pixel. Each mipmap has to be half the width and height of the previous
	level. At least one pixel will be always kept.*/
	virtual void regenerateMipMapLevels() = 0;

	//! Get name of texture (in most cases this is the filename)
	const io::SNamedPath& getName() const { return NamedPath; }

	E_RENDERABLE_TYPE getRenderableType() const {return ERT_TEXTURE;}

protected:

	//! Helper function, helps to get the desired texture creation format from the flags.
	/** \return Either ETCF_ALWAYS_32_BIT, ETCF_ALWAYS_16_BIT,
	ETCF_OPTIMIZED_FOR_QUALITY, or ETCF_OPTIMIZED_FOR_SPEED. */
	inline E_TEXTURE_CREATION_FLAG getTextureFormatFromFlags(u32 flags)
	{
		if (flags & ETCF_OPTIMIZED_FOR_SPEED)
			return ETCF_OPTIMIZED_FOR_SPEED;
		if (flags & ETCF_ALWAYS_16_BIT)
			return ETCF_ALWAYS_16_BIT;
		if (flags & ETCF_ALWAYS_32_BIT)
			return ETCF_ALWAYS_32_BIT;
		if (flags & ETCF_OPTIMIZED_FOR_QUALITY)
			return ETCF_OPTIMIZED_FOR_QUALITY;
		return ETCF_OPTIMIZED_FOR_SPEED;
	}

	io::SNamedPath NamedPath;
};


} // end namespace video
} // end namespace irr

#endif

