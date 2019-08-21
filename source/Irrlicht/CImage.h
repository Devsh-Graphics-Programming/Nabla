// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_H_INCLUDED__
#define __C_IMAGE_H_INCLUDED__

#include "IImage.h"
#include "rect.h"

namespace irr
{
namespace video
{

//! IImage implementation with a lot of special image operations for
//! 16 bit A1R5G5B5/32 Bit A8R8G8B8 images, which are used by the SoftwareDevice.
class CImage : public IImage
{
protected:
	//! destructor
	virtual ~CImage();

public:
	//! constructor from raw image data
	/** \param useForeignMemory: If true, the image will use the data pointer
	directly and own it from now on, which means it will also try to delete [] the
	data when the image will be destructed. If false, the memory will by copied. */
	CImage(asset::E_FORMAT format, const core::dimension2d<uint32_t>& size,
		void* data, bool ownForeignMemory=true);

	//! constructor for empty image
	CImage(asset::E_FORMAT format, const core::dimension2d<uint32_t>& size);

	//! .
	virtual void* getData() {return Data;}
	virtual const void* getData() const {return Data;}

	//! Returns width and height of image data.
	virtual const core::dimension2d<uint32_t>& getDimension() const;

	//! Returns bits per pixel.
	virtual uint32_t getBitsPerPixel() const;

	//! Returns image data size in bytes
	virtual uint32_t getImageDataSizeInBytes() const;

	//! Returns image data size in pixels
	virtual uint32_t getImageDataSizeInPixels() const;

	//! returns mask for red value of a pixel
	virtual uint32_t getRedMask() const;

	//! returns mask for green value of a pixel
	virtual uint32_t getGreenMask() const;

	//! returns mask for blue value of a pixel
	virtual uint32_t getBlueMask() const;

	//! returns mask for alpha value of a pixel
	virtual uint32_t getAlphaMask() const;

	//! returns a pixel
	virtual SColor getPixel(uint32_t x, uint32_t y) const;

	//! returns the color format
	virtual asset::E_FORMAT getColorFormat() const;

	//! returns pitch of image
	virtual uint32_t getPitch() const { return Pitch; }
private:

	//! assumes format and size has been set and creates the rest
	void initData();

	uint8_t* Data;
	core::dimension2d<uint32_t> Size;
	uint32_t Pitch;
	asset::E_FORMAT Format;

	bool DeleteMemory;
};

} // end namespace video
} // end namespace irr


#endif

