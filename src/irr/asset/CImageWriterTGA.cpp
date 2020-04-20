// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CImageWriterTGA.h"

#ifdef _IRR_COMPILE_WITH_TGA_WRITER_

#include "CImageLoaderTGA.h"
#include "IWriteFile.h"
#include "irr/asset/format/convertColor.h"
#include "irr/asset/ICPUImageView.h"
#include "os.h"

#include "os.h"

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

template<asset::E_FORMAT outFormat>
core::smart_refctd_ptr<asset::ICPUImage> getTGAConvertedOutput(const asset::ICPUImage* image)
{
	static_assert(!asset::isBlockCompressionFormat<outFormat>(), "Only non BC formats supported!");

	using CONVERSION_FILTER = asset::CConvertFormatImageFilter<asset::EF_UNKNOWN, outFormat>;

	core::smart_refctd_ptr<asset::ICPUImage> newConvertedImage;
	{
		auto referenceImageParams = image->getCreationParameters();
		auto referenceBuffer = image->getBuffer();
		auto referenceRegions = image->getRegions();
		auto referenceRegion = referenceRegions.begin();
		const auto newTexelOrBlockByteSize = asset::getTexelOrBlockBytesize(outFormat);

		asset::TexelBlockInfo referenceBlockInfo(referenceImageParams.format);
		core::vector3du32_SIMD referenceTrueExtent = referenceBlockInfo.convertTexelsToBlocks(referenceRegion->getTexelStrides());

		auto newImageParams = referenceImageParams;
		auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceTrueExtent.X * referenceTrueExtent.Y * referenceTrueExtent.Z * newTexelOrBlockByteSize);
		auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1);
		auto newRegion = newRegions->begin();
		*newRegion = *referenceRegion;

		newImageParams.format = outFormat;
		newConvertedImage = ICPUImage::create(std::move(newImageParams));
		newConvertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

		CONVERSION_FILTER convertFilter;
		CONVERSION_FILTER::state_type state;

		auto attachedRegion = newConvertedImage->getRegions().begin();

		state.inImage = image;
		state.outImage = newConvertedImage.get();
		state.inOffset = { 0, 0, 0 };
		state.inBaseLayer = 0;
		state.outOffset = { 0, 0, 0 };
		state.outBaseLayer = 0;
		state.extent = attachedRegion->getExtent();
		state.layerCount = attachedRegion->imageSubresource.layerCount;
		state.inMipLevel = attachedRegion->imageSubresource.mipLevel;
		state.outMipLevel = attachedRegion->imageSubresource.mipLevel;

		if (!convertFilter.execute(&state))
			os::Printer::log("Something went wrong while converting!", ELL_WARNING);

		return newConvertedImage;
	}
}

bool CImageWriterTGA::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

	SAssetWriteContext ctx{ _params, _file };

	const asset::ICPUImageView* imageView = IAsset::castDown<ICPUImageView>(_params.rootAsset);

	io::IWriteFile* file = _override->getOutputFile(_file, ctx, { imageView, 0u });

	core::smart_refctd_ptr<ICPUImage> convertedImage;
	{
		const auto channelCount = asset::getFormatChannelCount(imageView->getCreationParameters().format);
		if (channelCount == 1)
			convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8_SRGB>(imageView);
		else if (channelCount == 2)
			convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_A1R5G5B5_UNORM_PACK16>(imageView);
		else if(channelCount == 3)
			convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8G8B8_SRGB>(imageView);
		else
			convertedImage = IImageAssetHandlerBase::createImageDataForCommonWriting<asset::EF_R8G8B8A8_SRGB>(imageView);
	}
	
	const auto& convertedImageParams = convertedImage->getCreationParameters();
	const auto& convertedRegion = convertedImage->getRegions().begin();
	auto convertedFormat = convertedImageParams.format;

	asset::TexelBlockInfo blockInfo(convertedFormat);
	core::vector3du32_SIMD trueExtent = blockInfo.convertTexelsToBlocks(convertedRegion->getTexelStrides());

	core::vector3d<uint32_t> dim;
	dim.X = trueExtent.X;
	dim.Y = trueExtent.Y;
	dim.Z = trueExtent.Z;

	STGAHeader imageHeader;
	imageHeader.IdLength = 0;
	imageHeader.ColorMapType = 0;
	imageHeader.ImageType = (convertedFormat == EF_R8_SRGB) ? 3 : 2;
	imageHeader.FirstEntryIndex[0] = 0;
	imageHeader.FirstEntryIndex[1] = 0;
	imageHeader.ColorMapLength = 0;
	imageHeader.ColorMapEntrySize = 0;
	imageHeader.XOrigin[0] = 0;
	imageHeader.XOrigin[1] = 0;
	imageHeader.YOrigin[0] = 0;
	imageHeader.YOrigin[1] = 0;
	imageHeader.ImageWidth = trueExtent.X;
	imageHeader.ImageHeight = trueExtent.Y;

	// top left of image is the top. the image loader needs to
	// be fixed to only swap/flip
	imageHeader.ImageDescriptor = 1;
	
	switch (convertedFormat)
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

	uint8_t* scan_lines = (uint8_t*)convertedImage->getBuffer()->getPointer();
	if (!scan_lines)
		return false;

	// size of one pixel in bits
	uint32_t pixel_size_bits = convertedImage->getBytesPerPixel().getIntegerApprox();

	// length of one row of the source image in bytes
	uint32_t row_stride = (pixel_size_bits * imageHeader.ImageWidth);

	// length of one output row in bytes
	int32_t row_size = ((imageHeader.PixelDepth / 8) * imageHeader.ImageWidth);

	// allocate a row do translate data into
	auto rowPointerBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(row_size);
	auto row_pointer = reinterpret_cast<uint8_t*>(rowPointerBuffer->getPointer());

	uint32_t y;
	for (y = 0; y < imageHeader.ImageHeight; ++y)
	{
		memcpy(row_pointer, &scan_lines[y * row_stride], row_size);
		
		if (file->write(row_pointer, row_size) != row_size)
			break;
	}
	
	STGAExtensionArea extension;
	extension.ExtensionSize = sizeof(extension);
	extension.Gamma = isSRGBFormat(convertedFormat) ? ((100.0f / 30.0f) - 1.1f) : 1.0f;
	
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