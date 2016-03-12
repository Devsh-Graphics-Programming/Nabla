// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_H_INCLUDED__
#define __I_IMAGE_H_INCLUDED__

#include "IReferenceCounted.h"
#include "position2d.h"
#include "rect.h"
#include "SColor.h"

namespace irr
{
namespace video
{

//! Interface for software image data.
/** Image loaders create these images from files. IVideoDrivers convert
these images into their (hardware) textures.
*/
class IImage : public virtual IReferenceCounted
{
public:

	//! Lock function. Use this to get a pointer to the image data.
	/** After you don't need the pointer anymore, you must call unlock().
	\return Pointer to the image data. What type of data is pointed to
	depends on the color format of the image. For example if the color
	format is ECF_A8R8G8B8, it is of u32. Be sure to call unlock() after
	you don't need the pointer any more. */
	virtual void* lock() = 0;

	//! Unlock function.
	/** Should be called after the pointer received by lock() is not
	needed anymore. */
	virtual void unlock() = 0;

	//! Returns width and height of image data.
	virtual const core::dimension2d<u32>& getDimension() const = 0;

	//! Returns bits per pixel.
	virtual u32 getBitsPerPixel() const = 0;

	//! Returns image data size in bytes
	virtual u32 getImageDataSizeInBytes() const = 0;

	//! Returns image data size in pixels
	virtual u32 getImageDataSizeInPixels() const = 0;

	//! Returns a pixel
	virtual SColor getPixel(u32 x, u32 y) const = 0;

	//! Sets a pixel
	virtual void setPixel(u32 x, u32 y, const SColor &color, bool blend = false ) = 0;

	//! Returns the color format
	virtual ECOLOR_FORMAT getColorFormat() const = 0;

	//! Returns mask for red value of a pixel
	virtual u32 getRedMask() const = 0;

	//! Returns mask for green value of a pixel
	virtual u32 getGreenMask() const = 0;

	//! Returns mask for blue value of a pixel
	virtual u32 getBlueMask() const = 0;

	//! Returns mask for alpha value of a pixel
	virtual u32 getAlphaMask() const = 0;

	//! Returns pitch of image
	virtual u32 getPitch() const =0;

	//! Copies the image into the target, scaling the image to fit
	virtual void copyToScaling(void* target, u32 width, u32 height, ECOLOR_FORMAT format=ECF_A8R8G8B8, u32 pitch=0) =0;

	//! Copies the image into the target, scaling the image to fit
	virtual void copyToScaling(IImage* target) =0;

	//! copies this surface into another
	virtual void copyTo(IImage* target, const core::position2d<s32>& pos=core::position2d<s32>(0,0)) =0;

	//! copies this surface into another
	virtual void copyTo(IImage* target, const core::position2d<s32>& pos, const core::rect<s32>& sourceRect, const core::rect<s32>* clipRect=0) =0;

	//! copies this surface into another, using the alpha mask and cliprect and a color to add with
	virtual void copyToWithAlpha(IImage* target, const core::position2d<s32>& pos,
			const core::rect<s32>& sourceRect, const SColor &color,
			const core::rect<s32>* clipRect = 0) =0;

	//! copies this surface into another, scaling it to fit, appyling a box filter
	virtual void copyToScalingBoxFilter(IImage* target, s32 bias = 0, bool blend = false) = 0;

	//! fills the surface with given color
	virtual void fill(const SColor &color) =0;

	//! get the amount of Bits per Pixel of the given color format
	static u32 getBitsPerPixelFromFormat(const ECOLOR_FORMAT format)
	{
		switch(format)
		{
		case ECF_A1R5G5B5:
			return 16;
		case ECF_R5G6B5:
			return 16;
		case ECF_R8G8B8:
			return 24;
		case ECF_A8R8G8B8:
			return 32;
		case ECF_R16F:
			return 16;
		case ECF_G16R16F:
			return 32;
		case ECF_A16B16G16R16F:
			return 64;
		case ECF_R32F:
			return 32;
		case ECF_G32R32F:
			return 64;
		case ECF_A32B32G32R32F:
			return 128;
		case ECF_R8:
			return 8;
		case ECF_R8G8:
			return 16;
        case ECF_R8G8B8A8:
            return 32;
        case ECF_RGB_BC1:
        case ECF_RGBA_BC1:
            return 4;
        case ECF_RGBA_BC2:
        case ECF_RGBA_BC3:
            return 8;
        case ECF_R_BC4:
            return 4;
        case ECF_RG_BC5:
            return 8;
        case ECF_8BIT_PIX:
			return 8;
		case ECF_16BIT_PIX:
			return 16;
		case ECF_24BIT_PIX:
			return 24;
		case ECF_32BIT_PIX:
			return 32;
		case ECF_48BIT_PIX: // rgb @ 16bit
            return 48;
        case ECF_64BIT_PIX:
			return 64;
		case ECF_96BIT_PIX:
			return 96;
		case ECF_128BIT_PIX:
            return 128;
        case ECF_DEPTH16:
            return 16;
        case ECF_DEPTH24:
            return 24;
        case ECF_DEPTH32F:
        case ECF_DEPTH24_STENCIL8:
            return 32;
        case ECF_DEPTH32F_STENCIL8:
            return 40;
		default:
			return 0;
		}
	}

};

} // end namespace video
} // end namespace irr

#endif

