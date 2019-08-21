// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "irr/asset/format/convertColor.h"

#include "CImage.h"

namespace irr
{
namespace video
{

//! Constructor of empty image
CImage::CImage(asset::E_FORMAT format, const core::dimension2d<uint32_t>& size)
:Data(0), Size(size), Format(format), DeleteMemory(true)
{
	initData();
}


//! Constructor from raw data
CImage::CImage(asset::E_FORMAT format, const core::dimension2d<uint32_t>& size, void* data,
			bool ownForeignMemory)
: Data(0), Size(size), Format(format), DeleteMemory(!ownForeignMemory)
{
	if (ownForeignMemory)
	{
		Data = (uint8_t*)0xdeadbeefu;
		initData();
		Data = (uint8_t*)data;
	}
	else
	{
		Data = 0;
		initData();
		memcpy(Data, data, Size.Height * Pitch);
	}
}


//! assumes format and size has been set and creates the rest
void CImage::initData()
{
#ifdef _IRR_DEBUG
	setDebugName("CImage");
#endif

	// Pitch should be aligned...
	Pitch = getBitsPerPixel() * Size.Width;

	Pitch /= 8;

	if (!Data)
	{
		Data = new uint8_t[Size.Height * Pitch];
		DeleteMemory=true;
	}
}


//! destructor
CImage::~CImage()
{
	if ( DeleteMemory )
		delete [] Data;
}


//! Returns width and height of image data.
const core::dimension2d<uint32_t>& CImage::getDimension() const
{
	return Size;
}


//! Returns bits per pixel.
uint32_t CImage::getBitsPerPixel() const
{
	return getBitsPerPixelFromFormat(Format);
}

//! Returns image data size in bytes
uint32_t CImage::getImageDataSizeInBytes() const
{
	return (getBitsPerPixel() * Size.Width * Size.Height)/8;
}


//! Returns image data size in pixels
uint32_t CImage::getImageDataSizeInPixels() const
{
	return Size.Width * Size.Height;
}


//! returns mask for red value of a pixel
uint32_t CImage::getRedMask() const
{
	switch(Format)
	{
	case asset::EF_A1R5G5B5_UNORM_PACK16:
		return 0x1F<<10;
	case asset::EF_B5G6R5_UNORM_PACK16:
		return 0x1F<<11;
	case asset::EF_R8G8B8_UNORM:
		return 0x00FF0000;
	case asset::EF_B8G8R8A8_UNORM:
		return 0x00FF0000;
	case asset::EF_R8G8B8A8_UNORM:
		return 0xFF000000;
	default:
		return 0x0;
	}
}


//! returns mask for green value of a pixel
uint32_t CImage::getGreenMask() const
{
	switch(Format)
	{
	case asset::EF_A1R5G5B5_UNORM_PACK16:
		return 0x1F<<5;
	case asset::EF_B5G6R5_UNORM_PACK16:
		return 0x3F<<5;
	case asset::EF_R8G8B8_UNORM:
		return 0x0000FF00;
	case asset::EF_B8G8R8A8_UNORM:
		return 0x0000FF00;
	case asset::EF_R8G8B8A8_UNORM:
		return 0x00FF0000;
	default:
		return 0x0;
	}
}


//! returns mask for blue value of a pixel
uint32_t CImage::getBlueMask() const
{
	switch(Format)
	{
	case asset::EF_A1R5G5B5_UNORM_PACK16:
		return 0x1F;
	case asset::EF_B5G6R5_UNORM_PACK16:
		return 0x1F;
	case asset::EF_R8G8B8_UNORM:
		return 0x000000FF;
	case asset::EF_B8G8R8A8_UNORM:
		return 0x000000FF;
	case asset::EF_R8G8B8A8_UNORM:
		return 0x0000FF00;
	default:
		return 0x0;
	}
}


//! returns mask for alpha value of a pixel
uint32_t CImage::getAlphaMask() const
{
	switch(Format)
	{
	case asset::EF_A1R5G5B5_UNORM_PACK16:
		return 0x1<<15;
	case asset::EF_B5G6R5_UNORM_PACK16:
		return 0x0;
	case asset::EF_R8G8B8_UNORM:
		return 0x0;
	case asset::EF_B8G8R8A8_UNORM:
		return 0xFF000000;
	case asset::EF_R8G8B8A8_UNORM:
		return 0x000000FF;
	default:
		return 0x0;
	}
}


//! returns a pixel
SColor CImage::getPixel(uint32_t x, uint32_t y) const
{
	if (x >= Size.Width || y >= Size.Height || isDepthOrStencilFormat(Format) || isBlockCompressionFormat(Format))
		return SColor(0);

    uint32_t color8888 = 0u;
    const void* original[4]{};
	switch(Format)
	{
	case asset::EF_A1R5G5B5_UNORM_PACK16:
    {
        double decOutput[4];
        original[0] = &reinterpret_cast<uint16_t*>(Data)[y*Size.Width + x];
        decodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, double>(original, decOutput, 0u, 0u);
        uint64_t encInput[4];
        std::transform(decOutput, decOutput+4, encInput, [](double x) { return x*255.; });
        encodePixels<asset::EF_B8G8R8A8_UINT, uint64_t>(&color8888, encInput);
        return color8888;
    }
	case asset::EF_B5G6R5_UNORM_PACK16:
    {
        double decOutput[4];
        original[0] = &reinterpret_cast<uint16_t*>(Data)[y*Size.Width + x];
        decodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, double>(original, decOutput, 0u, 0u);
        decOutput[3] = 1.;
        uint64_t encInput[4];
        std::transform(decOutput, decOutput+4, encInput, [](double x) { return x*255.; });
        encodePixels<asset::EF_B8G8R8A8_UINT, uint64_t>(&color8888, encInput);
        return color8888;
    }
	case asset::EF_B8G8R8A8_UNORM:
		return reinterpret_cast<uint32_t*>(Data)[y*Size.Width + x];
	case asset::EF_R8G8B8_UNORM:
	{
        original[0] = Data+(y*3)*Size.Width + (x*3);
        convertColor<asset::EF_R8G8B8_UINT, asset::EF_B8G8R8A8_UINT>(original, &color8888, 0u, 0u);
        reinterpret_cast<uint8_t*>(&color8888)[3] = 0xffu;
        return color8888;
	}
	default:
        return SColor(0u);
	}
}


//! returns the color format
asset::E_FORMAT CImage::getColorFormat() const
{
	return Format;
}


} // end namespace video
} // end namespace irr
