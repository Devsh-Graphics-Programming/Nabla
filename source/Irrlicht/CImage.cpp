// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImage.h"
#include "irrString.h"
#include "CColorConverter.h"
#include "CBlit.h"

namespace irr
{
namespace video
{

//! Constructor of empty image
CImage::CImage(ECOLOR_FORMAT format, const core::dimension2d<uint32_t>& size)
:Data(0), Size(size), Format(format), DeleteMemory(true)
{
	initData();
}


//! Constructor from raw data
CImage::CImage(ECOLOR_FORMAT format, const core::dimension2d<uint32_t>& size, void* data,
			bool ownForeignMemory, bool deleteForeignMemory)
: Data(0), Size(size), Format(format), DeleteMemory(deleteForeignMemory)
{
	if (ownForeignMemory)
	{
		Data = (uint8_t*)0xbadf00d;
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
#ifdef _DEBUG
	setDebugName("CImage");
#endif
	BitsPerPixel = getBitsPerPixelFromFormat(Format);

	// Pitch should be aligned...
	Pitch = BitsPerPixel * Size.Width;

	Pitch /= 8;

	if (!Data)
	{
		DeleteMemory=true;
		Data = new uint8_t[Size.Height * Pitch];
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
	return BitsPerPixel;
}

//! Returns image data size in bytes
uint32_t CImage::getImageDataSizeInBytes() const
{
	return (BitsPerPixel * Size.Width * Size.Height)/8;
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
	case ECF_A1R5G5B5:
		return 0x1F<<10;
	case ECF_R5G6B5:
		return 0x1F<<11;
	case ECF_R8G8B8:
		return 0x00FF0000;
	case ECF_A8R8G8B8:
		return 0x00FF0000;
	case ECF_R8G8B8A8:
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
	case ECF_A1R5G5B5:
		return 0x1F<<5;
	case ECF_R5G6B5:
		return 0x3F<<5;
	case ECF_R8G8B8:
		return 0x0000FF00;
	case ECF_A8R8G8B8:
		return 0x0000FF00;
	case ECF_R8G8B8A8:
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
	case ECF_A1R5G5B5:
		return 0x1F;
	case ECF_R5G6B5:
		return 0x1F;
	case ECF_R8G8B8:
		return 0x000000FF;
	case ECF_A8R8G8B8:
		return 0x000000FF;
	case ECF_R8G8B8A8:
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
	case ECF_A1R5G5B5:
		return 0x1<<15;
	case ECF_R5G6B5:
		return 0x0;
	case ECF_R8G8B8:
		return 0x0;
	case ECF_A8R8G8B8:
		return 0xFF000000;
	case ECF_R8G8B8A8:
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
		case ECF_A1R5G5B5:
		{
			uint16_t * dest = (uint16_t*) (Data + ( y * Pitch ) + ( x << 1 ));
			*dest = video::A8R8G8B8toA1R5G5B5( color.color );
		} break;

		case ECF_R5G6B5:
		{
			uint16_t * dest = (uint16_t*) (Data + ( y * Pitch ) + ( x << 1 ));
			*dest = video::A8R8G8B8toR5G6B5( color.color );
		} break;

		case ECF_R8G8B8:
		{
			uint8_t* dest = Data + ( y * Pitch ) + ( x * 3 );
			dest[0] = (uint8_t)color.getRed();
			dest[1] = (uint8_t)color.getGreen();
			dest[2] = (uint8_t)color.getBlue();
		} break;

		case ECF_A8R8G8B8:
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
#ifndef _DEBUG
		default:
			break;
#endif
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
	if (x >= Size.Width || y >= Size.Height || (Format>=video::ECF_RGB_BC1&&Format<=video::ECF_DEPTH32F_STENCIL8) )
		return SColor(0);

	switch(Format)
	{
	case ECF_A1R5G5B5:
		return A1R5G5B5toA8R8G8B8(((uint16_t*)Data)[y*Size.Width + x]);
	case ECF_R5G6B5:
		return R5G6B5toA8R8G8B8(((uint16_t*)Data)[y*Size.Width + x]);
	case ECF_A8R8G8B8:
		return ((uint32_t*)Data)[y*Size.Width + x];
	case ECF_R8G8B8:
		{
			uint8_t* p = Data+(y*3)*Size.Width + (x*3);
			return SColor(255,p[0],p[1],p[2]);
		}
	default:
        return SColor(0);
	}
}


//! returns the color format
ECOLOR_FORMAT CImage::getColorFormat() const
{
	return Format;
}


//! copies this surface into another at given position
void CImage::copyTo(IImage* target, const core::position2d<int32_t>& pos)
{
    if (!target)
        return;

    if ((target->getColorFormat()>=video::ECF_RGB_BC1&&target->getColorFormat()<=video::ECF_RG_BC5)||(Format>=video::ECF_RGB_BC1&&Format<=video::ECF_RG_BC5))
        return;

	Blit(BLITTER_TEXTURE, target, 0, &pos, this, 0, 0);
}


//! copies this surface partially into another at given position
void CImage::copyTo(IImage* target, const core::position2d<int32_t>& pos, const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect)
{
    if (!target)
        return;

    if ((target->getColorFormat()>=video::ECF_RGB_BC1&&target->getColorFormat()<=video::ECF_RG_BC5)||(Format>=video::ECF_RGB_BC1&&Format<=video::ECF_RG_BC5))
        return;

	Blit(BLITTER_TEXTURE, target, clipRect, &pos, this, &sourceRect, 0);
}


//! copies this surface into another, using the alpha mask, a cliprect and a color to add with
void CImage::copyToWithAlpha(IImage* target, const core::position2d<int32_t>& pos, const core::rect<int32_t>& sourceRect, const SColor &color, const core::rect<int32_t>* clipRect)
{
    if (!target)
        return;

    if ((target->getColorFormat()>=video::ECF_RGB_BC1&&target->getColorFormat()<=video::ECF_RG_BC5)||(Format>=video::ECF_RGB_BC1&&Format<=video::ECF_RG_BC5))
        return;

	// color blend only necessary on not full spectrum aka. color.color != 0xFFFFFFFF
	Blit(color.color == 0xFFFFFFFF ? BLITTER_TEXTURE_ALPHA_BLEND: BLITTER_TEXTURE_ALPHA_COLOR_BLEND,
			target, clipRect, &pos, this, &sourceRect, color.color);
}


//! copies this surface into another, scaling it to the target image size
// note: this is very very slow.
void CImage::copyToScaling(void* target, uint32_t width, uint32_t height, ECOLOR_FORMAT format, uint32_t pitch)
{
	if (!target || !width || !height)
		return;

    /// we don't want to support block compression XD - even if we wanted.. PATENTS!
    if ((format>=video::ECF_RGB_BC1&&format<=video::ECF_RG_BC5)||(Format>=video::ECF_RGB_BC1&&Format<=video::ECF_RG_BC5))
        return;

	const uint32_t bpp=getBitsPerPixelFromFormat(format);
	if (0==pitch)
		pitch = (width*bpp)/8;

	if (Format==format && Size.Width==width && Size.Height==height)
	{
		if (pitch==Pitch)
		{
			memcpy(target, Data, height*pitch);
			return;
		}
		else
		{
			uint8_t* tgtpos = (uint8_t*) target;
			uint8_t* srcpos = Data;
			const uint32_t bwidth = (width*bpp)/8;
			const uint32_t rest = pitch-bwidth;
			for (uint32_t y=0; y<height; ++y)
			{
				// copy scanline
				memcpy(tgtpos, srcpos, bwidth);
				// clear pitch
				memset(tgtpos+bwidth, 0, rest);
				tgtpos += pitch;
				srcpos += Pitch;
			}
			return;
		}
	}

    /// there aint no interpolation && scaling for depth
    /// and we dont know the bit && channel layout for bit-width pixels
    if (format>=ECF_8BIT_PIX&&format<=ECF_UNKNOWN)
        return;

    if (format==ECF_R8||format==ECF_R8G8)
    {
		//irr::os::Printer::log("DevSH will support conversion from and to GL_R8 and GL_R8G8 when he has absolutely nothing else to do.", ELL_ERROR);
        return;
    }

	const float sourceXStep = (float)Size.Width / (float)width;
	const float sourceYStep = (float)Size.Height / (float)height;
	int32_t yval=0, syval=0;
	float sy = 0.0f;
	for (uint32_t y=0; y<height; ++y)
	{
		float sx = 0.0f;
		for (uint32_t x=0; x<width; ++x)
		{
			CColorConverter::convert_viaFormat(Data+ syval + (((int32_t)sx)*BitsPerPixel)/8, Format, 1, ((uint8_t*)target)+ yval + (x*bpp)/8, format);
			sx+=sourceXStep;
		}
		sy+=sourceYStep;
		syval=((int32_t)sy)*Pitch;
		yval+=pitch;
	}
}


//! copies this surface into another, scaling it to the target image size
// note: this is very very slow.
void CImage::copyToScaling(IImage* target)
{
	if (!target)
		return;

	const core::dimension2d<uint32_t>& targetSize = target->getDimension();

	if (targetSize==Size)
	{
		copyTo(target);
		return;
	}

	copyToScaling(target->lock(), targetSize.Width, targetSize.Height, target->getColorFormat());
	target->unlock();
}


//! copies this surface into another, scaling it to fit it.
void CImage::copyToScalingBoxFilter(IImage* target, int32_t bias, bool blend)
{
    if (!target)
        return;

    if ((target->getColorFormat()>=video::ECF_RGB_BC1&&target->getColorFormat()<=video::ECF_RG_BC5)||(Format>=video::ECF_RGB_BC1&&Format<=video::ECF_RG_BC5))
        return;

	const core::dimension2d<uint32_t> destSize = target->getDimension();

	const float sourceXStep = (float) Size.Width / (float) destSize.Width;
	const float sourceYStep = (float) Size.Height / (float) destSize.Height;

	target->lock();

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

	target->unlock();
}


//! fills the surface with given color
void CImage::fill(const SColor &color)
{
	uint32_t c;

	switch ( Format )
	{
		case ECF_A1R5G5B5:
			c = color.toA1R5G5B5();
			c |= c << 16;
			break;
		case ECF_R5G6B5:
			c = video::A8R8G8B8toR5G6B5( color.color );
			c |= c << 16;
			break;
		case ECF_A8R8G8B8:
			c = color.color;
			break;
		case ECF_R8G8B8:
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

    if (Format<video::ECF_RGB_BC1||Format>video::ECF_RG_BC5)
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
