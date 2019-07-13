// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CColorConverter.h"
#include "SColor.h"
#include "os.h"
#include "string.h"

namespace irr
{
namespace video
{

//! converts a monochrome bitmap to A1R5G5B5 data
void CColorConverter::convert1BitTo16Bit(const uint8_t* in, int16_t* out, int32_t width, int32_t height, int32_t linepad, bool flip)
{
	if (!in || !out)
		return;

	if (flip)
		out += width * height;

	for (int32_t y=0; y<height; ++y)
	{
		int32_t shift = 7;
		if (flip)
			out -= width;

		for (int32_t x=0; x<width; ++x)
		{
			out[x] = *in>>shift & 0x01 ? (int16_t)0xffff : (int16_t)0x8000;

			if ((--shift)<0) // 8 pixel done
			{
				shift=7;
				++in;
			}
		}

		if (shift != 7) // width did not fill last byte
			++in;

		if (!flip)
			out += width;
		in += linepad;
	}
}



//! converts a 4 bit palettized image to A1R5G5B5
void CColorConverter::convert4BitTo16Bit(const uint8_t* in, int16_t* out, int32_t width, int32_t height, const int32_t* palette, int32_t linepad, bool flip)
{
	if (!in || !out || !palette)
		return;

	if (flip)
		out += width*height;

	for (int32_t y=0; y<height; ++y)
	{
		int32_t shift = 4;
		if (flip)
			out -= width;

		for (int32_t x=0; x<width; ++x)
		{
			out[x] = X8R8G8B8toA1R5G5B5(palette[(uint8_t)((*in >> shift) & 0xf)]);

			if (shift==0)
			{
				shift = 4;
				++in;
			}
			else
				shift = 0;
		}

		if (shift == 0) // odd width
			++in;

		if (!flip)
			out += width;
		in += linepad;
	}
}



//! converts a 8 bit palettized image into A1R5G5B5
void CColorConverter::convert8BitTo16Bit(const uint8_t* in, int16_t* out, int32_t width, int32_t height, const int32_t* palette, int32_t linepad, bool flip)
{
	if (!in || !out || !palette)
		return;

	if (flip)
		out += width * height;

	for (int32_t y=0; y<height; ++y)
	{
		if (flip)
			out -= width; // one line back
		for (int32_t x=0; x<width; ++x)
		{
			out[x] = X8R8G8B8toA1R5G5B5(palette[(uint8_t)(*in)]);
			++in;
		}
		if (!flip)
			out += width;
		in += linepad;
	}
}

//! converts a 8 bit palettized or non palettized image (A8) into R8G8B8
void CColorConverter::convert8BitTo24Bit(const uint8_t* in, uint8_t* out, int32_t width, int32_t height, const uint8_t* palette, int32_t linepad, bool flip)
{
	if (!in || !out )
		return;

	const int32_t lineWidth = 3 * width;
	if (flip)
		out += lineWidth * height;

	for (int32_t y=0; y<height; ++y)
	{
		if (flip)
			out -= lineWidth; // one line back
		for (int32_t x=0; x< lineWidth; x += 3)
		{
			if ( palette )
			{
				out[x+0] = palette[ (in[0] << 2 ) + 2];
				out[x+1] = palette[ (in[0] << 2 ) + 1];
				out[x+2] = palette[ (in[0] << 2 ) + 0];
			}
			else
			{
				out[x+0] = in[0];
				out[x+1] = in[0];
				out[x+2] = in[0];
			}
			++in;
		}
		if (!flip)
			out += lineWidth;
		in += linepad;
	}
}

//! converts a 8 bit palettized or non palettized image (A8) into R8G8B8
void CColorConverter::convert8BitTo32Bit(const uint8_t* in, uint8_t* out, int32_t width, int32_t height, const uint8_t* palette, int32_t linepad, bool flip)
{
	if (!in || !out )
		return;

	const uint32_t lineWidth = 4 * width;
	if (flip)
		out += lineWidth * height;

	uint32_t x;
	register uint32_t c;
	for (uint32_t y=0; y < (uint32_t) height; ++y)
	{
		if (flip)
			out -= lineWidth; // one line back

		if ( palette )
		{
			for (x=0; x < (uint32_t) width; x += 1)
			{
				c = in[x];
				((uint32_t*)out)[x] = ((uint32_t*)palette)[ c ];
			}
		}
		else
		{
			for (x=0; x < (uint32_t) width; x += 1)
			{
				c = in[x];

				((uint32_t*)out)[x] = 0xFF000000 | c << 16 | c << 8 | c;
			}

		}

		if (!flip)
			out += lineWidth;
		in += width + linepad;
	}
}



//! converts 16bit data to 16bit data
void CColorConverter::convert16BitTo16Bit(const int16_t* in, int16_t* out, int32_t width, int32_t height, int32_t linepad, bool flip)
{
	if (!in || !out)
		return;

	if (flip)
		out += width * height;

	for (int32_t y=0; y<height; ++y)
	{
		if (flip)
			out -= width;

		memcpy(out, in, width*sizeof(int16_t));

		if (!flip)
			out += width;
		in += width;
		in += linepad;
	}
}



//! copies R8G8B8 24bit data to 24bit data
void CColorConverter::convert24BitTo24Bit(const uint8_t* in, uint8_t* out, int32_t width, int32_t height, int32_t linepad, bool flip, bool bgr)
{
	if (!in || !out)
		return;

	const int32_t lineWidth = 3 * width;
	if (flip)
		out += lineWidth * height;

	for (int32_t y=0; y<height; ++y)
	{
		if (flip)
			out -= lineWidth;
		if (bgr)
		{
			for (int32_t x=0; x<lineWidth; x+=3)
			{
				out[x+0] = in[x+2];
				out[x+1] = in[x+1];
				out[x+2] = in[x+0];
			}
		}
		else
		{
			memcpy(out,in,lineWidth);
		}
		if (!flip)
			out += lineWidth;
		in += lineWidth;
		in += linepad;
	}
}



//! Resizes the surface to a new size and converts it at the same time
//! to an A8R8G8B8 format, returning the pointer to the new buffer.
void CColorConverter::convert16bitToA8R8G8B8andResize(const int16_t* in, int32_t* out, int32_t newWidth, int32_t newHeight, int32_t currentWidth, int32_t currentHeight)
{
	if (!newWidth || !newHeight)
		return;

	// note: this is very very slow. (i didn't want to write a fast version.
	// but hopefully, nobody wants to convert surfaces every frame.

	float sourceXStep = (float)currentWidth / (float)newWidth;
	float sourceYStep = (float)currentHeight / (float)newHeight;
	float sy;
	int32_t t;

	for (int32_t x=0; x<newWidth; ++x)
	{
		sy = 0.0f;

		for (int32_t y=0; y<newHeight; ++y)
		{
			t = in[(int32_t)(((int32_t)sy)*currentWidth + x*sourceXStep)];
			t = (((t >> 15)&0x1)<<31) |	(((t >> 10)&0x1F)<<19) |
				(((t >> 5)&0x1F)<<11) |	(t&0x1F)<<3;
			out[(int32_t)(y*newWidth + x)] = t;

			sy+=sourceYStep;
		}
	}
}



//! copies X8R8G8B8 32 bit data
void CColorConverter::convert32BitTo32Bit(const int32_t* in, int32_t* out, int32_t width, int32_t height, int32_t linepad, bool flip)
{
	if (!in || !out)
		return;

	if (flip)
		out += width * height;

	for (int32_t y=0; y<height; ++y)
	{
		if (flip)
			out -= width;

		memcpy(out, in, width*sizeof(int32_t));

		if (!flip)
			out += width;
		in += width;
		in += linepad;
	}
}



void CColorConverter::convert_A1R5G5B5toR8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint8_t * dB = (uint8_t *)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		dB[2] = (*sB & 0x7c00) >> 7;
		dB[1] = (*sB & 0x03e0) >> 2;
		dB[0] = (*sB & 0x1f) << 3;

		sB += 1;
		dB += 3;
	}
}

void CColorConverter::convert_A1R5G5B5toB8G8R8(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint8_t * dB = (uint8_t *)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		dB[0] = (*sB & 0x7c00) >> 7;
		dB[1] = (*sB & 0x03e0) >> 2;
		dB[2] = (*sB & 0x1f) << 3;

		sB += 1;
		dB += 3;
	}
}

void CColorConverter::convert_A1R5G5B5toA8R8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint32_t* dB = (uint32_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
		*dB++ = A1R5G5B5toA8R8G8B8(*sB++);
}

void CColorConverter::convert_A1R5G5B5toA1R5G5B5(const void* sP, int32_t sN, void* dP)
{
	memcpy(dP, sP, sN * 2);
}

void CColorConverter::convert_A1R5G5B5toR5G6B5(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint16_t* dB = (uint16_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
		*dB++ = A1R5G5B5toR5G6B5(*sB++);
}

void CColorConverter::convert_A8R8G8B8toR8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint8_t* sB = (uint8_t*)sP;
	uint8_t* dB = (uint8_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		// sB[3] is alpha
		dB[0] = sB[2];
		dB[1] = sB[1];
		dB[2] = sB[0];

		sB += 4;
		dB += 3;
	}
}

void CColorConverter::convert_A8R8G8B8toB8G8R8(const void* sP, int32_t sN, void* dP)
{
	uint8_t* sB = (uint8_t*)sP;
	uint8_t* dB = (uint8_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		// sB[3] is alpha
		dB[0] = sB[0];
		dB[1] = sB[1];
		dB[2] = sB[2];

		sB += 4;
		dB += 3;
	}
}

void CColorConverter::convert_A8R8G8B8toA8R8G8B8(const void* sP, int32_t sN, void* dP)
{
	memcpy(dP, sP, sN * 4);
}

void CColorConverter::convert_A8R8G8B8toA1R5G5B5(const void* sP, int32_t sN, void* dP)
{
	uint32_t* sB = (uint32_t*)sP;
	uint16_t* dB = (uint16_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
		*dB++ = A8R8G8B8toA1R5G5B5(*sB++);
}

void CColorConverter::convert_A8R8G8B8toR5G6B5(const void* sP, int32_t sN, void* dP)
{
	uint8_t * sB = (uint8_t *)sP;
	uint16_t* dB = (uint16_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		int32_t r = sB[2] >> 3;
		int32_t g = sB[1] >> 2;
		int32_t b = sB[0] >> 3;

		dB[0] = (r << 11) | (g << 5) | (b);

		sB += 4;
		dB += 1;
	}
}

void CColorConverter::convert_A8R8G8B8toR3G3B2(const void* sP, int32_t sN, void* dP)
{
	uint8_t* sB = (uint8_t*)sP;
	uint8_t* dB = (uint8_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		uint8_t r = sB[2] & 0xe0;
		uint8_t g = (sB[1] & 0xe0) >> 3;
		uint8_t b = (sB[0] & 0xc0) >> 6;

		dB[0] = (r | g | b);

		sB += 4;
		dB += 1;
	}
}

void CColorConverter::convert_R8G8B8toR8G8B8(const void* sP, int32_t sN, void* dP)
{
	memcpy(dP, sP, sN * 3);
}

void CColorConverter::convert_R8G8B8toA8R8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint8_t*  sB = (uint8_t* )sP;
	uint32_t* dB = (uint32_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		*dB = 0xff000000 | (sB[0]<<16) | (sB[1]<<8) | sB[2];

		sB += 3;
		++dB;
	}
}

void CColorConverter::convert_R8G8B8toA1R5G5B5(const void* sP, int32_t sN, void* dP)
{
	uint8_t * sB = (uint8_t *)sP;
	uint16_t* dB = (uint16_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		int32_t r = sB[0] >> 3;
		int32_t g = sB[1] >> 3;
		int32_t b = sB[2] >> 3;

		dB[0] = (0x8000) | (r << 10) | (g << 5) | (b);

		sB += 3;
		dB += 1;
	}
}

void CColorConverter::convert_B8G8R8toA8R8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint8_t*  sB = (uint8_t* )sP;
	uint32_t* dB = (uint32_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		*dB = 0xff000000 | (sB[2]<<16) | (sB[1]<<8) | sB[0];

		sB += 3;
		++dB;
	}
}

void CColorConverter::convert_B8G8R8A8toA8R8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint8_t* sB = (uint8_t*)sP;
	uint8_t* dB = (uint8_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		dB[0] = sB[3];
		dB[1] = sB[2];
		dB[2] = sB[1];
		dB[3] = sB[0];

		sB += 4;
		dB += 4;
	}

}

void CColorConverter::convert_R8G8B8toR5G6B5(const void* sP, int32_t sN, void* dP)
{
	uint8_t * sB = (uint8_t *)sP;
	uint16_t* dB = (uint16_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		int32_t r = sB[0] >> 3;
		int32_t g = sB[1] >> 2;
		int32_t b = sB[2] >> 3;

		dB[0] = (r << 11) | (g << 5) | (b);

		sB += 3;
		dB += 1;
	}
}

void CColorConverter::convert_R5G6B5toR5G6B5(const void* sP, int32_t sN, void* dP)
{
	memcpy(dP, sP, sN * 2);
}

void CColorConverter::convert_R5G6B5toR8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint8_t * dB = (uint8_t *)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		dB[0] = (*sB & 0xf800) >> 8;
		dB[1] = (*sB & 0x07e0) >> 3;
		dB[2] = (*sB & 0x001f) << 3;

		sB += 1;
		dB += 3;
	}
}

void CColorConverter::convert_R5G6B5toB8G8R8(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint8_t * dB = (uint8_t *)dP;

	for (int32_t x = 0; x < sN; ++x)
	{
		dB[2] = (*sB & 0xf800) >> 8;
		dB[1] = (*sB & 0x07e0) >> 3;
		dB[0] = (*sB & 0x001f) << 3;

		sB += 1;
		dB += 3;
	}
}

void CColorConverter::convert_R5G6B5toA8R8G8B8(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint32_t* dB = (uint32_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
		*dB++ = R5G6B5toA8R8G8B8(*sB++);
}

void CColorConverter::convert_R5G6B5toA1R5G5B5(const void* sP, int32_t sN, void* dP)
{
	uint16_t* sB = (uint16_t*)sP;
	uint16_t* dB = (uint16_t*)dP;

	for (int32_t x = 0; x < sN; ++x)
		*dB++ = R5G6B5toA1R5G5B5(*sB++);
}


void CColorConverter::convert_viaFormat(const void* sP, asset::E_FORMAT sF, int32_t sN,
				void* dP, asset::E_FORMAT dF)
{
    using namespace asset;
	switch (sF)
	{
		case EF_A1R5G5B5_UNORM_PACK16:
			switch (dF)
			{
				case EF_A1R5G5B5_UNORM_PACK16:
					convert_A1R5G5B5toA1R5G5B5(sP, sN, dP);
				break;
				case EF_B5G6R5_UNORM_PACK16:
					convert_A1R5G5B5toR5G6B5(sP, sN, dP);
				break;
				case EF_B8G8R8A8_UNORM:
					convert_A1R5G5B5toA8R8G8B8(sP, sN, dP);
				break;
				case EF_R8G8B8_UNORM:
					convert_A1R5G5B5toR8G8B8(sP, sN, dP);
				break;
				default:
					break;
			}
		break;
		case EF_B5G6R5_UNORM_PACK16:
			switch (dF)
			{
				case EF_A1R5G5B5_UNORM_PACK16:
					convert_R5G6B5toA1R5G5B5(sP, sN, dP);
				break;
				case EF_B5G6R5_UNORM_PACK16:
					convert_R5G6B5toR5G6B5(sP, sN, dP);
				break;
				case EF_B8G8R8A8_UNORM:
					convert_R5G6B5toA8R8G8B8(sP, sN, dP);
				break;
				case EF_R8G8B8_UNORM:
					convert_R5G6B5toR8G8B8(sP, sN, dP);
				break;
				default:
					break;
			}
		break;
		case EF_B8G8R8A8_UNORM:
			switch (dF)
			{
				case EF_A1R5G5B5_UNORM_PACK16:
					convert_A8R8G8B8toA1R5G5B5(sP, sN, dP);
				break;
				case EF_B5G6R5_UNORM_PACK16:
					convert_A8R8G8B8toR5G6B5(sP, sN, dP);
				break;
				case EF_B8G8R8A8_UNORM:
					convert_A8R8G8B8toA8R8G8B8(sP, sN, dP);
				break;
				case EF_R8G8B8_UNORM:
					convert_A8R8G8B8toR8G8B8(sP, sN, dP);
				break;
				default:
					break;
			}
		break;
		case EF_R8G8B8_UNORM:
			switch (dF)
			{
				case EF_A1R5G5B5_UNORM_PACK16:
					convert_R8G8B8toA1R5G5B5(sP, sN, dP);
				break;
				case EF_B5G6R5_UNORM_PACK16:
					convert_R8G8B8toR5G6B5(sP, sN, dP);
				break;
				case EF_B8G8R8A8_UNORM:
					convert_R8G8B8toA8R8G8B8(sP, sN, dP);
				break;
				case EF_R8G8B8_UNORM:
					convert_R8G8B8toR8G8B8(sP, sN, dP);
				break;
				default:
					break;
			}
		break;
        default: break;
	}
}


} // end namespace video
} // end namespace irr
