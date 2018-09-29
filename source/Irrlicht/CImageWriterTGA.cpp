// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageWriterTGA.h"

#ifdef _IRR_COMPILE_WITH_TGA_WRITER_

#include "CImageLoaderTGA.h"
#include "IWriteFile.h"
#include "CColorConverter.h"
#include "ICPUTexture.h"

namespace irr
{
namespace video
{

CImageWriterTGA::CImageWriterTGA()
{
#ifdef _DEBUG
	setDebugName("CImageWriterTGA");
#endif
}

bool CImageWriterTGA::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    const video::CImageData* image =
#   ifndef _DEBUG
        static_cast<const video::CImageData*>(_params.rootAsset);
#   else
        dynamic_cast<const video::CImageData*>(_params.rootAsset);
#   endif
    assert(image);

	STGAHeader imageHeader;
	imageHeader.IdLength = 0;
	imageHeader.ColorMapType = 0;
	imageHeader.ImageType = 2;
	imageHeader.FirstEntryIndex[0] = 0;
	imageHeader.FirstEntryIndex[1] = 0;
	imageHeader.ColorMapLength = 0;
	imageHeader.ColorMapEntrySize = 0;
	imageHeader.XOrigin[0] = 0;
	imageHeader.XOrigin[1] = 0;
	imageHeader.YOrigin[0] = 0;
	imageHeader.YOrigin[1] = 0;
	imageHeader.ImageWidth = image->getSize().X;
	imageHeader.ImageHeight = image->getSize().Y;

	// top left of image is the top. the image loader needs to
	// be fixed to only swap/flip
	imageHeader.ImageDescriptor = (1 << 5);

   // chances are good we'll need to swizzle data, so i'm going
	// to convert and write one scan line at a time. it's also
	// a bit cleaner this way
	void (*CColorConverter_convertFORMATtoFORMAT)(const void*, int32_t, void*) = 0;
	switch(image->getColorFormat())
	{
	case ECF_A8R8G8B8:
		CColorConverter_convertFORMATtoFORMAT
			= CColorConverter::convert_A8R8G8B8toA8R8G8B8;
		imageHeader.PixelDepth = 32;
		imageHeader.ImageDescriptor |= 8;
		break;
	case ECF_A1R5G5B5:
		CColorConverter_convertFORMATtoFORMAT
			= CColorConverter::convert_A1R5G5B5toA1R5G5B5;
		imageHeader.PixelDepth = 16;
		imageHeader.ImageDescriptor |= 1;
		break;
	case ECF_R5G6B5:
		CColorConverter_convertFORMATtoFORMAT
			= CColorConverter::convert_R5G6B5toA1R5G5B5;
		imageHeader.PixelDepth = 16;
		imageHeader.ImageDescriptor |= 1;
		break;
	case ECF_R8G8B8:
		CColorConverter_convertFORMATtoFORMAT
			= CColorConverter::convert_R8G8B8toR8G8B8;
		imageHeader.PixelDepth = 24;
		imageHeader.ImageDescriptor |= 0;
		break;
#ifndef _DEBUG
	default:
		break;
#endif
	}

	// couldn't find a color converter
	if (!CColorConverter_convertFORMATtoFORMAT)
		return false;

	if (_file->write(&imageHeader, sizeof(imageHeader)) != sizeof(imageHeader))
		return false;

	uint8_t* scan_lines = (uint8_t*)image->getData();
	if (!scan_lines)
		return false;

	// size of one pixel in bits
	uint32_t pixel_size_bits = image->getBitsPerPixel();

	// length of one row of the source image in bytes
	uint32_t row_stride = (pixel_size_bits * imageHeader.ImageWidth)/8;

	// length of one output row in bytes
	int32_t row_size = ((imageHeader.PixelDepth / 8) * imageHeader.ImageWidth);

	// allocate a row do translate data into
	uint8_t* row_pointer = new uint8_t[row_size];

	uint32_t y;
	for (y = 0; y < imageHeader.ImageHeight; ++y)
	{
		// source, length [pixels], destination
		if (image->getColorFormat()==ECF_R8G8B8)
			CColorConverter::convert24BitTo24Bit(&scan_lines[y * row_stride], row_pointer, imageHeader.ImageWidth, 1, 0, 0, true);
		else
			CColorConverter_convertFORMATtoFORMAT(&scan_lines[y * row_stride], imageHeader.ImageWidth, row_pointer);
		if (_file->write(row_pointer, row_size) != row_size)
			break;
	}

	delete [] row_pointer;

	STGAFooter imageFooter;
	imageFooter.ExtensionOffset = 0;
	imageFooter.DeveloperOffset = 0;
	strncpy(imageFooter.Signature, "TRUEVISION-XFILE.", 18);

	if (_file->write(&imageFooter, sizeof(imageFooter)) < (int32_t)sizeof(imageFooter))
		return false;

	return imageHeader.ImageHeight <= y;
}

} // namespace video
} // namespace irr

#endif

