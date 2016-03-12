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
#include "matrix4.h"

namespace irr
{
namespace video
{


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

//! Enum for the mode for texture locking. Read-Only, write-only or read/write.
enum E_TEXTURE_LOCK_MODE
{
	//! The default mode. Texture can be read and written to.
	ETLM_READ_WRITE = 0,

	//! Read only. The texture is downloaded, but not uploaded again.
	/** Often used to read back shader generated textures. */
	ETLM_READ_ONLY,

	//! Write only. The texture is not downloaded and might be uninitialised.
	/** The updated texture is uploaded to the GPU.
	Used for initialising the shader from the CPU. */
	ETLM_WRITE_ONLY
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
class ITexture : public virtual IRenderable
{
public:

	//! constructor
	ITexture(const io::path& name) : NamedPath(name)
	{
	}

	//! Get dimension (=size) of the texture.
	/** \return The size of the texture. */
	virtual const core::dimension2d<u32>& getSize() const = 0;

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

