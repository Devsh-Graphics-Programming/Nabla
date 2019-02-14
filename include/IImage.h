// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_H_INCLUDED__
#define __I_IMAGE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "rect.h"
#include "SColor.h"

namespace irr
{
namespace video
{

//! Interface for software image data.
/** Image loaders create these images from files.
*/
class IImage : public virtual core::IReferenceCounted
{
public:

	//! Use this to get a pointer to the image data.
	/**
	\return Pointer to the image data. What type of data is pointed to
	depends on the color format of the image. For example if the color
	format is EF_B8G8R8A8_UNORM, it is of uint32_t. */
	virtual void* getData() = 0;
	virtual const void* getData() const = 0;

	//! Returns width and height of image data.
	virtual const core::dimension2d<uint32_t>& getDimension() const = 0;

	//! Returns bits per pixel.
	virtual uint32_t getBitsPerPixel() const = 0;

	//! Returns image data size in bytes
	virtual uint32_t getImageDataSizeInBytes() const = 0;

	//! Returns image data size in pixels
	virtual uint32_t getImageDataSizeInPixels() const = 0;

	//! Returns a pixel
	virtual SColor getPixel(uint32_t x, uint32_t y) const = 0;

	//! Sets a pixel
	virtual void setPixel(uint32_t x, uint32_t y, const SColor &color, bool blend = false ) = 0;

	//! Returns the color format
	virtual asset::E_FORMAT getColorFormat() const = 0;

	//! Returns mask for red value of a pixel
	virtual uint32_t getRedMask() const = 0;

	//! Returns mask for green value of a pixel
	virtual uint32_t getGreenMask() const = 0;

	//! Returns mask for blue value of a pixel
	virtual uint32_t getBlueMask() const = 0;

	//! Returns mask for alpha value of a pixel
	virtual uint32_t getAlphaMask() const = 0;

	//! Returns pitch of image
	virtual uint32_t getPitch() const =0;

	//! copies this surface into another
	virtual void copyTo(IImage* target, const core::position2d<int32_t>& pos=core::position2d<int32_t>(0,0)) =0;

	//! copies this surface into another
	virtual void copyTo(IImage* target, const core::position2d<int32_t>& pos, const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect=0) =0;

	//! copies this surface into another, using the alpha mask and cliprect and a color to add with
	virtual void copyToWithAlpha(IImage* target, const core::position2d<int32_t>& pos,
			const core::rect<int32_t>& sourceRect, const SColor &color,
			const core::rect<int32_t>* clipRect = 0) =0;

	//! copies this surface into another, scaling it to fit, appyling a box filter
	virtual void copyToScalingBoxFilter(IImage* target, int32_t bias = 0, bool blend = false) = 0;

	//! fills the surface with given color
	virtual void fill(const SColor &color) =0;
};

} // end namespace video
} // end namespace irr

#endif

