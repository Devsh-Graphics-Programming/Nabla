// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageWriterTGA.h"

#ifdef _IRR_COMPILE_WITH_TGA_WRITER_

#include "CImageLoaderTGA.h"
#include "IWriteFile.h"
#include "irr/video/convertColor.h"
#include "irr/asset/ICPUTexture.h"

namespace irr
{
namespace asset
{

CImageWriterTGA::CImageWriterTGA()
{
#ifdef _IRR_DEBUG
	setDebugName("CImageWriterTGA");
#endif
}

bool CImageWriterTGA::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext ctx{_params, _file};

    const asset::CImageData* image =
#   ifndef _IRR_DEBUG
        static_cast<const asset::CImageData*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::CImageData*>(_params.rootAsset);
#   endif
    assert(image);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, { image, 0u });
	
	auto format = image->getColorFormat();

	STGAHeader imageHeader;
	imageHeader.IdLength = 0;
	imageHeader.ColorMapType = 0;
	imageHeader.ImageType = ((format == EF_R8_SRGB) || (format == EF_R8_UNORM)) ? 3 : 2;
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
	imageHeader.ImageDescriptor = 1;
	
	switch (format)
	{
		case asset::EF_R8G8B8A8_SRGB:
			{
				imageHeader.PixelDepth = 32;
				imageHeader.ImageDescriptor |= 8;
			}
			break;
		case asset::EF_R8G8B8_SRGB:
			{
				imageHeader.PixelDepth = 24;
				imageHeader.ImageDescriptor |= 0;
			}
			break;
		case asset::EF_A1R5G5B5_UNORM_PACK16:
			{
				imageHeader.PixelDepth = 16;
				imageHeader.ImageDescriptor |= 1;
			}
			break;
		case asset::EF_R8_SRGB:
		case asset::EF_R8_UNORM:
			{
				imageHeader.PixelDepth = 8;
				imageHeader.ImageDescriptor |= 0;
			}
			break;
		default:
			{
				os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
				return false;
			}
	}

	if (file->write(&imageHeader, sizeof(imageHeader)) != sizeof(imageHeader))
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

	auto size = image->getSize();

	uint32_t y;
	for (y = 0; y < imageHeader.ImageHeight; ++y)
	{
		switch (format) {
			case asset::EF_R8_SRGB:
			case asset::EF_A1R5G5B5_UNORM_PACK16:
				{
					memcpy(row_pointer, &scan_lines[y * row_stride], imageHeader.ImageWidth * (imageHeader.PixelDepth / 8));
				}
			break;
			
			case asset::EF_R8_UNORM:
				{
					const void *src_container[4] = {&scan_lines[y * row_stride], nullptr, nullptr, nullptr};
					video::convertColor<EF_R8_UNORM, EF_R8_SRGB>(src_container, row_pointer, imageHeader.ImageWidth, size);
				}
			break;
			
			// EF_R8G8B8(A8)_SRGB would need swizzling to EF_B8G8R8(A8)_SRGB.
			case asset::EF_R8G8B8_SRGB:
				{
					const void *src_container[4] = {&scan_lines[y * row_stride], nullptr, nullptr, nullptr};
					video::convertColor<EF_R8G8B8_SRGB, EF_B8G8R8_SRGB>(src_container, row_pointer, imageHeader.ImageWidth, size);
				}
			break;
			
			case asset::EF_R8G8B8A8_SRGB:
				{
					const void *src_container[4] = {&scan_lines[y * row_stride], nullptr, nullptr, nullptr};
					video::convertColor<EF_R8G8B8A8_SRGB, EF_B8G8R8A8_SRGB>(src_container, row_pointer, imageHeader.ImageWidth, size);
				}
			break;
			
			default:
			{
				os::Printer::log("Unsupported color format, operation aborted.", ELL_ERROR);
				if (row_pointer) delete [] row_pointer;
				return false;
			}
		}
		
		if (file->write(row_pointer, row_size) != row_size)
			break;
	}
	
	delete [] row_pointer;
	
	STGAExtensionArea extension;
	extension.ExtensionSize = sizeof(extension);
	extension.Gamma = isSRGBFormat(format) ? ((100.0f / 30.0f) - 1.1f) : 1.0f;
	
	STGAFooter imageFooter;
	imageFooter.ExtensionOffset = _file->getPos();
	imageFooter.DeveloperOffset = 0;
	strncpy(imageFooter.Signature, "TRUEVISION-XFILE.", 18);
	
	if (file->write(&extension, sizeof(extension)) < (int32_t)sizeof(extension))
		return false;

	if (file->write(&imageFooter, sizeof(imageFooter)) < (int32_t)sizeof(imageFooter))
		return false;

	return imageHeader.ImageHeight <= y;
}

} // namespace video
} // namespace irr

#endif

