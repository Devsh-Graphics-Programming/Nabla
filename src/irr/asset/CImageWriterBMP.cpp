// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageWriterBMP.h"

#ifdef _IRR_COMPILE_WITH_BMP_WRITER_

#include "CImageLoaderBMP.h"
#include "IWriteFile.h"
#include "CColorConverter.h"
#include "irr/asset/ICPUTexture.h"

namespace irr
{
namespace asset
{

CImageWriterBMP::CImageWriterBMP()
{
#ifdef _IRR_DEBUG
	setDebugName("CImageWriterBMP");
#endif
}

bool CImageWriterBMP::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext ctx{_params, _file};

	// we always write 24-bit color because nothing really reads 32-bit
    const asset::CImageData* image =
#   ifndef _IRR_DEBUG
        static_cast<const asset::CImageData*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::CImageData*>(_params.rootAsset);
#   endif
    assert(image);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {image, 0u});

	SBMPHeader imageHeader;
	imageHeader.Id = 0x4d42;
	imageHeader.Reserved = 0;
	imageHeader.BitmapDataOffset = sizeof(imageHeader);
	imageHeader.BitmapHeaderSize = 0x28;
	imageHeader.Width = image->getSize().X;
	imageHeader.Height = image->getSize().Y;
	imageHeader.Planes = 1;
	imageHeader.BPP = 24;
	imageHeader.Compression = 0;
	imageHeader.PixelPerMeterX = 0;
	imageHeader.PixelPerMeterY = 0;
	imageHeader.Colors = 0;
	imageHeader.ImportantColors = 0;

	// data size is rounded up to next larger 4 bytes boundary
	imageHeader.BitmapDataSize = imageHeader.Width * imageHeader.BPP / 8;
	imageHeader.BitmapDataSize = (imageHeader.BitmapDataSize + 3) & ~3;
	imageHeader.BitmapDataSize *= imageHeader.Height;

	// file size is data size plus offset to data
	imageHeader.FileSize = imageHeader.BitmapDataOffset + imageHeader.BitmapDataSize;

	// bitmaps are stored upside down and padded so we always do this
	void (*CColorConverter_convertFORMATtoFORMAT)(const void*, int32_t, void*) = 0;
	switch(image->getColorFormat())
	{
	case asset::EF_R8G8B8_UNORM:
		CColorConverter_convertFORMATtoFORMAT
			= video::CColorConverter::convert_R8G8B8toR8G8B8;
		break;
	case asset::EF_B8G8R8A8_UNORM:
		CColorConverter_convertFORMATtoFORMAT
			= video::CColorConverter::convert_A8R8G8B8toB8G8R8;
		break;
	case asset::EF_A1R5G5B5_UNORM_PACK16:
		CColorConverter_convertFORMATtoFORMAT
			= video::CColorConverter::convert_A1R5G5B5toR8G8B8;
		break;
	case asset::EF_B5G6R5_UNORM_PACK16:
		CColorConverter_convertFORMATtoFORMAT
			= video::CColorConverter::convert_R5G6B5toR8G8B8;
		break;
	default:
		break;
	}

	// couldn't find a color converter
	if (!CColorConverter_convertFORMATtoFORMAT)
		return false;

	// write the bitmap header
	if (file->write(&imageHeader, sizeof(imageHeader)) != sizeof(imageHeader))
		return false;

	uint8_t* scan_lines = (uint8_t*)image->getData();
	if (!scan_lines)
		return false;

	// size of one pixel in bits
	uint32_t pixel_size_bits = image->getBitsPerPixel();

	// length of one row of the source image in bytes
	uint32_t row_stride = (pixel_size_bits * imageHeader.Width)/8;

	// length of one row in bytes, rounded up to nearest 4-byte boundary
	int32_t row_size = ((3 * imageHeader.Width) + 3) & ~3;

	// allocate and clear memory for our scan line
	uint8_t* row_pointer = new uint8_t[row_size];
	memset(row_pointer, 0, row_size);

	// convert the image to 24-bit BGR and flip it over
	int32_t y;
	for (y = imageHeader.Height - 1; 0 <= y; --y)
	{
		if (image->getColorFormat()==asset::EF_R8G8B8_UNORM)
            video::CColorConverter::convert24BitTo24Bit(&scan_lines[y * row_stride], row_pointer, imageHeader.Width, 1, 0, false, true);
		else
			// source, length [pixels], destination
			CColorConverter_convertFORMATtoFORMAT(&scan_lines[y * row_stride], imageHeader.Width, row_pointer);
		if (file->write(row_pointer, row_size) < row_size)
			break;
	}

	// clean up our scratch area
	delete [] row_pointer;

	return y < 0;
}

} // namespace asset
} // namespace irr

#endif

