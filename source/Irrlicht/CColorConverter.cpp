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


void CColorConverter::convert_R8G8B8toR8G8B8(const void* sP, int32_t sN, void* dP)
{
	memcpy(dP, sP, sN * 3);
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

} // end namespace video
} // end namespace irr
