// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImage.h"
#include "CColorConverter.h"
#include "CBlit.h"
#include "irr/video/convertColor.h"

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


//! sets a pixel
void CImage::setPixel(uint32_t x, uint32_t y, const SColor &color, bool blend)
{
	if (x >= Size.Width || y >= Size.Height)
		return;

	switch(Format)
	{
		case asset::EF_A1R5G5B5_UNORM_PACK16:
		{
			uint16_t * dest = (uint16_t*) (Data + ( y * Pitch ) + ( x << 1 ));
			*dest = video::A8R8G8B8toA1R5G5B5( color.color );
		} break;

		case asset::EF_B5G6R5_UNORM_PACK16:
		{
			uint16_t * dest = (uint16_t*) (Data + ( y * Pitch ) + ( x << 1 ));
			*dest = video::A8R8G8B8toR5G6B5( color.color );
		} break;

		case asset::EF_R8G8B8_UNORM:
		{
			uint8_t* dest = Data + ( y * Pitch ) + ( x * 3 );
			dest[0] = (uint8_t)color.getRed();
			dest[1] = (uint8_t)color.getGreen();
			dest[2] = (uint8_t)color.getBlue();
		} break;

		case asset::EF_B8G8R8A8_UNORM:
		{
			uint32_t * dest = (uint32_t*) (Data + ( y * Pitch ) + ( x << 2 ));
//			*dest = blend ? PixelBlend32 ( *dest, color.color ) : color.color;
			if (!blend)
				*dest = color.color;
			else
			{
				uint32_t p = PixelBlend32(*dest, color.color );	// baw sodan
				((uint8_t*)(&p))[3] = 255u-(255u-color.getAlpha())*(255u-((*dest)>>24))/255;
				*dest = p;
			}
		} break;
		default:
			break;
	}
}


//! sets a pixel
// baw
/*void CImage::setPixel(uint32_t x, uint32_t y, const SColor &color)
{
	if (x >= Size.Width || y >= Size.Height)
		return;

	uint32_t *dest = (uint32_t*) (Data + ( y * Pitch ) + ( x << 2 ));
	uint32_t p = PixelBlend32(*dest, color.color );
	p |= 0xff000000;
	*dest = p;
}*/


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


//! copies this surface into another at given position
void CImage::copyTo(IImage* target, const core::position2d<int32_t>& pos)
{
    if (!target)
        return;

    if (asset::isBlockCompressionFormat(target->getColorFormat()) || asset::isBlockCompressionFormat(Format))
        return;

	Blit(BLITTER_TEXTURE, target, 0, &pos, this, 0, 0);
}


//! copies this surface partially into another at given position
void CImage::copyTo(IImage* target, const core::position2d<int32_t>& pos, const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect)
{
    if (!target)
        return;

    if (asset::isBlockCompressionFormat(target->getColorFormat()) || asset::isBlockCompressionFormat(Format))
        return;

	Blit(BLITTER_TEXTURE, target, clipRect, &pos, this, &sourceRect, 0);
}


//! copies this surface into another, using the alpha mask, a cliprect and a color to add with
void CImage::copyToWithAlpha(IImage* target, const core::position2d<int32_t>& pos, const core::rect<int32_t>& sourceRect, const SColor &color, const core::rect<int32_t>* clipRect)
{
    if (!target)
        return;

    if (asset::isBlockCompressionFormat(target->getColorFormat()) || asset::isBlockCompressionFormat(Format))
        return;

	// color blend only necessary on not full spectrum aka. color.color != 0xFFFFFFFF
	Blit(color.color == 0xFFFFFFFF ? BLITTER_TEXTURE_ALPHA_BLEND: BLITTER_TEXTURE_ALPHA_COLOR_BLEND,
			target, clipRect, &pos, this, &sourceRect, color.color);
}


//! copies this surface into another, scaling it to fit it.
void CImage::copyToScalingBoxFilter(IImage* target, int32_t bias, bool blend)
{
    if (!target)
        return;

    if (asset::isBlockCompressionFormat(target->getColorFormat()) || asset::isBlockCompressionFormat(Format))
        return;

	const core::dimension2d<uint32_t> destSize = target->getDimension();

	const float sourceXStep = (float) Size.Width / (float) destSize.Width;
	const float sourceYStep = (float) Size.Height / (float) destSize.Height;

	int32_t fx = core::ceil32( sourceXStep );
	int32_t fy = core::ceil32( sourceYStep );
	float sx;
	float sy;

	sy = 0.f;
	for ( uint32_t y = 0; y != destSize.Height; ++y )
	{
		sx = 0.f;
		for ( uint32_t x = 0; x != destSize.Width; ++x )
		{
			target->setPixel( x, y,
				getPixelBox( core::floor32(sx), core::floor32(sy), fx, fy, bias ), blend );
			sx += sourceXStep;
		}
		sy += sourceYStep;
	}
}


//! fills the surface with given color
void CImage::fill(const SColor &color)
{
	uint32_t c;

	switch ( Format )
	{
		case asset::EF_A1R5G5B5_UNORM_PACK16:
			c = color.toA1R5G5B5();
			c |= c << 16;
			break;
		case asset::EF_B5G6R5_UNORM_PACK16:
			c = video::A8R8G8B8toR5G6B5( color.color );
			c |= c << 16;
			break;
		case asset::EF_B8G8R8A8_UNORM:
			c = color.color;
			break;
		case asset::EF_R8G8B8_UNORM:
		{
			uint8_t rgb[3];
			CColorConverter::convert_A8R8G8B8toR8G8B8(&color, 1, rgb);
			const uint32_t size = getImageDataSizeInBytes();
			for (uint32_t i=0; i<size; i+=3)
			{
				memcpy(Data+i, rgb, 3);
			}
			return;
		}
		break;
		default:
		// TODO: Handle other formats
			return;
	}
	memset32( Data, c, getImageDataSizeInBytes() );
}


//! get a filtered pixel
inline SColor CImage::getPixelBox( int32_t x, int32_t y, int32_t fx, int32_t fy, int32_t bias ) const
{
	SColor c;
	int32_t a = 0, r = 0, g = 0, b = 0;

    if (!asset::isBlockCompressionFormat(Format))
    {
        for ( int32_t dx = 0; dx != fx; ++dx )
        {
            for ( int32_t dy = 0; dy != fy; ++dy )
            {
                c = getPixel(	core::s32_min ( x + dx, Size.Width - 1 ) ,
                                core::s32_min ( y + dy, Size.Height - 1 )
                            );

                a += c.getAlpha();
                r += c.getRed();
                g += c.getGreen();
                b += c.getBlue();
            }

        }

        int32_t sdiv = s32_log2_s32(fx * fy);

        a = core::s32_clamp( ( a >> sdiv ) + bias, 0, 255 );
        r = core::s32_clamp( ( r >> sdiv ) + bias, 0, 255 );
        g = core::s32_clamp( ( g >> sdiv ) + bias, 0, 255 );
        b = core::s32_clamp( ( b >> sdiv ) + bias, 0, 255 );
    }

	c.set( a, r, g, b );
	return c;
}


} // end namespace video
} // end namespace irr
