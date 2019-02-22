// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_COLOR_CONVERTER_H_INCLUDED__
#define __C_COLOR_CONVERTER_H_INCLUDED__

#include "irr/core/Types.h"
#include "irr/macros.h"
#include "SColor.h"

namespace irr
{
namespace video
{

class CColorConverter
{
    CColorConverter() = delete;
public:

	//! converts a monochrome bitmap to A1R5G5B5
	static void convert1BitTo16Bit(const uint8_t* in, int16_t* out, int32_t width, int32_t height, int32_t linepad=0, bool flip=false);

	//! converts a 4 bit palettized image to A1R5G5B5
	static void convert4BitTo16Bit(const uint8_t* in, int16_t* out, int32_t width, int32_t height, const int32_t* palette, int32_t linepad=0, bool flip=false);

	//! converts a 8 bit palettized image to A1R5G5B5
	static void convert8BitTo16Bit(const uint8_t* in, int16_t* out, int32_t width, int32_t height, const int32_t* palette, int32_t linepad=0, bool flip=false);

	//! converts a 8 bit palettized or non palettized image (A8) into R8G8B8
	static void convert8BitTo24Bit(const uint8_t* in, uint8_t* out, int32_t width, int32_t height, const uint8_t* palette, int32_t linepad = 0, bool flip=false);

	//! converts a 8 bit palettized or non palettized image (A8) into A8R8G8B8
	static void convert8BitTo32Bit(const uint8_t* in, uint8_t* out, int32_t width, int32_t height, const uint8_t* palette, int32_t linepad = 0, bool flip=false);

	//! converts R8G8B8 16 bit data to A1R5G5B5 data
	static void convert16BitTo16Bit(const int16_t* in, int16_t* out, int32_t width, int32_t height, int32_t linepad=0, bool flip=false);

	//! copies R8G8B8 24 bit data to 24 data, and flips and
	//! mirrors the image during the process.
	static void convert24BitTo24Bit(const uint8_t* in, uint8_t* out, int32_t width, int32_t height, int32_t linepad=0, bool flip=false, bool bgr=false);

	//! Resizes the surface to a new size and converts it at the same time
	//! to an A8R8G8B8 format, returning the pointer to the new buffer.
	static void convert16bitToA8R8G8B8andResize(const int16_t* in, int32_t* out, int32_t newWidth, int32_t newHeight, int32_t currentWidth, int32_t currentHeight);

	//! copies X8R8G8B8 32 bit data, and flips and
	//! mirrors the image during the process.
	static void convert32BitTo32Bit(const int32_t* in, int32_t* out, int32_t width, int32_t height, int32_t linepad, bool flip=false);


	//! functions for converting one image format to another efficiently
	//! and hopefully correctly.
	//!
	//! \param sP pointer to source pixel data
	//! \param sN number of source pixels to copy
	//! \param dP pointer to destination data buffer. must be big enough
	//! to hold sN pixels in the output format.
	static void convert_A1R5G5B5toR8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_A1R5G5B5toB8G8R8(const void* sP, int32_t sN, void* dP);
	static void convert_A1R5G5B5toA8R8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_A1R5G5B5toA1R5G5B5(const void* sP, int32_t sN, void* dP);
	static void convert_A1R5G5B5toR5G6B5(const void* sP, int32_t sN, void* dP);

	static void convert_A8R8G8B8toR8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_A8R8G8B8toB8G8R8(const void* sP, int32_t sN, void* dP);
	static void convert_A8R8G8B8toA8R8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_A8R8G8B8toA1R5G5B5(const void* sP, int32_t sN, void* dP);
	static void convert_A8R8G8B8toR5G6B5(const void* sP, int32_t sN, void* dP);

	static void convert_A8R8G8B8toR3G3B2(const void* sP, int32_t sN, void* dP);
	static void convert_R8G8B8toR8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_R8G8B8toA8R8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_R8G8B8toA1R5G5B5(const void* sP, int32_t sN, void* dP);
	static void convert_R8G8B8toR5G6B5(const void* sP, int32_t sN, void* dP);
	static void convert_B8G8R8toA8R8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_B8G8R8A8toA8R8G8B8(const void* sP, int32_t sN, void* dP);

	static void convert_R5G6B5toR5G6B5(const void* sP, int32_t sN, void* dP);
	static void convert_R5G6B5toR8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_R5G6B5toB8G8R8(const void* sP, int32_t sN, void* dP);
	static void convert_R5G6B5toA8R8G8B8(const void* sP, int32_t sN, void* dP);
	static void convert_R5G6B5toA1R5G5B5(const void* sP, int32_t sN, void* dP);
	static void convert_viaFormat(const void* sP, asset::E_FORMAT sF, int32_t sN,
				void* dP, asset::E_FORMAT dF);
};


} // end namespace video
} // end namespace irr

#endif

