// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CImageLoaderTGA.h"

#ifdef _NBL_COMPILE_WITH_TGA_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/ICPUImage.h"

#include "nbl/asset/interchange/IImageAssetHandlerBase.h"
#include "nbl/asset/filters/CConvertFormatImageFilter.h"

namespace nbl
{
namespace asset
{
	/*
		For color pallete stream. Create a buffer containing 
		a single row from taking an ICPUBuffer as a single 
		input row and convert it to any format.
	*/

	template<E_FORMAT inputFormat, E_FORMAT outputFormat>
	static inline core::smart_refctd_ptr<ICPUBuffer> createSingleRowBufferFromRawData(core::smart_refctd_ptr<asset::ICPUBuffer> inputBuffer)
	{
		auto inputTexelOrBlockByteSize = inputBuffer->getSize();
		const uint32_t texelOrBlockLength = inputTexelOrBlockByteSize / asset::getTexelOrBlockBytesize(inputFormat);

		auto outputBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(texelOrBlockLength * asset::getTexelOrBlockBytesize(outputFormat));
		auto outputTexelOrBlockByteSize = outputBuffer->getSize();

		for (auto i = 0ull; i < texelOrBlockLength; ++i)
		{
			const void* srcPix[] = { reinterpret_cast<const uint8_t*>(inputBuffer->getPointer()) + i * inputTexelOrBlockByteSize, nullptr, nullptr, nullptr };
			asset::convertColor<inputFormat, outputFormat>(srcPix, reinterpret_cast<uint8_t*>(outputBuffer->getPointer()) + i * outputTexelOrBlockByteSize, 0, 0);
		}

		return outputBuffer;
	}

//! loads a compressed tga.
void CImageLoaderTGA::loadCompressedImage(io::IReadFile *file, const STGAHeader& header, const uint32_t wholeSizeWithPitchInBytes, core::smart_refctd_ptr<ICPUBuffer>& bufferData) const
{
	// This was written and sent in by Jon Pry, thank you very much!
	// I only changed the formatting a little bit.
	int32_t bytesPerPixel = header.PixelDepth/8;
	int32_t imageSizeInBytes =  header.ImageHeight * header.ImageWidth * bytesPerPixel;
	bufferData = core::make_smart_refctd_ptr<ICPUBuffer>(wholeSizeWithPitchInBytes);
	auto data = reinterpret_cast<uint8_t*>(bufferData->getPointer());
	int32_t currentByte = 0;

	while(currentByte < imageSizeInBytes)
	{
		uint8_t chunkheader = 0;
		file->read(&chunkheader, sizeof(uint8_t)); // Read The Chunk's Header

		if(chunkheader < 128) // If The Chunk Is A 'RAW' Chunk
		{
			chunkheader++; // Add 1 To The Value To Get Total Number Of Raw Pixels

			file->read(&data[currentByte], bytesPerPixel * chunkheader);
			currentByte += bytesPerPixel * chunkheader;
		}
		else
		{
			// thnx to neojzs for some fixes with this code

			// If It's An RLE Header
			chunkheader -= 127; // Subtract 127 To Get Rid Of The ID Bit

			int32_t dataOffset = currentByte;
			file->read(&data[dataOffset], bytesPerPixel);

			currentByte += bytesPerPixel;

			for(int32_t counter = 1; counter < chunkheader; counter++)
			{
				for(int32_t elementCounter=0; elementCounter < bytesPerPixel; elementCounter++)
					data[currentByte + elementCounter] = data[dataOffset + elementCounter];

				currentByte += bytesPerPixel;
			}
		}
	}
}

//! returns true if the file maybe is able to be loaded by this class
bool CImageLoaderTGA::isALoadableFileFormat(io::IReadFile* _file) const
{
	if (!_file)
		return false;
	
    const size_t prevPos = _file->getPos();

	STGAFooter footer;
	memset(&footer, 0, sizeof(STGAFooter));
	_file->seek(_file->getSize() - sizeof(STGAFooter));
	_file->read(&footer, sizeof(STGAFooter));
	
	// 16 bytes for "TRUEVISION-XFILE", 17th byte is '.', and the 18th byte contains '\0'.
	if (strncmp(footer.Signature, "TRUEVISION-XFILE.", 18u) != 0)
	{
		os::Printer::log("Invalid (non-TGA) file!", ELL_ERROR);
		return false;
	}

	float gamma;

	if (footer.ExtensionOffset == 0)
	{
		os::Printer::log("Gamma information is not present! Assuming 2.333333", ELL_WARNING);
		gamma = 2.333333f;
	}
	else
	{
		STGAExtensionArea extension;
		_file->seek(footer.ExtensionOffset);
		_file->read(&extension, sizeof(STGAExtensionArea));
		
		gamma = extension.Gamma;
		
		if (gamma == 0.0f)
		{
			os::Printer::log("Gamma information is not present! Assuming 2.333333", ELL_WARNING);
			gamma = 2.333333f;
		}
		
		// TODO - pass gamma to LoadAsset()?
		// Actually I think metadata will be in used here in near future
	}
	
    _file->seek(prevPos);
	
	return true;
}

/*
	Targa formats needs two y-axis flips. The first is a flip to get the Y conforms to OpenGL coords.
	The second flip is defined from within the .tga file itself (header.ImageDescriptor & 0x20).
	if flip - perform two flips (OpenGL + Targa) = no flipping. Don't flip the image at all in that case
	if not - do an OpenGL flip
*/

template <E_FORMAT srcFormat, E_FORMAT destFormat>
core::smart_refctd_ptr<ICPUImage> createAndconvertImageData(ICPUImage::SCreationParams& imgInfo, core::smart_refctd_ptr<ICPUBuffer>&& texelBuffer, bool flip)
{
	static_assert((!asset::isBlockCompressionFormat<srcFormat>() || !asset::isBlockCompressionFormat<destFormat>()), "Only non BC formats supported!");

	core::smart_refctd_ptr<ICPUImage> newConvertedImage;
	core::smart_refctd_ptr<ICPUImage> inputCreationImage;
	{
		imgInfo.format = srcFormat;
		auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
		ICPUImage::SBufferCopy& region = regions->front();

		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.bufferOffset = 0u;
		region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(imgInfo.extent.width, asset::getTexelOrBlockBytesize(srcFormat));
		region.bufferImageHeight = 0u;
		region.imageOffset = { 0u, 0u, 0u };
		region.imageExtent = imgInfo.extent;

		inputCreationImage = asset::ICPUImage::create(std::move(imgInfo));
		inputCreationImage->setBufferAndRegions(std::move(texelBuffer), regions);
		
		bool OpenGlFlip = flip;
		if (OpenGlFlip)
			asset::IImageAssetHandlerBase::performImageFlip(inputCreationImage);
	}

	if (srcFormat == destFormat)
		newConvertedImage = inputCreationImage;
	else
	{
		using CONVERSION_FILTER = CConvertFormatImageFilter<srcFormat, destFormat>;
		CONVERSION_FILTER convertFilter;
		CONVERSION_FILTER::state_type state;
		{
			auto referenceImageParams = inputCreationImage->getCreationParameters();
			auto referenceBuffer = inputCreationImage->getBuffer();
			auto referenceRegions = inputCreationImage->getRegions();
			auto referenceRegion = referenceRegions.begin();
			const auto newTexelOrBlockByteSize = asset::getTexelOrBlockBytesize(destFormat);

			asset::TexelBlockInfo newBlockInfo(destFormat);
			core::vector3du32_SIMD newTrueExtent = newBlockInfo.convertTexelsToBlocks(referenceRegion->getTexelStrides());

			auto newImageParams = referenceImageParams;
			auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(newTrueExtent.X * newTrueExtent.Y * newTrueExtent.Z * newTexelOrBlockByteSize);
			auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1);
			newRegions->front() = *referenceRegion;

			newImageParams.format = destFormat;
			newConvertedImage = ICPUImage::create(std::move(newImageParams));
			newConvertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);
		}

		auto attachedRegion = newConvertedImage->getRegions().begin();

		state.inImage = inputCreationImage.get();
		state.outImage = newConvertedImage.get();
		state.inOffset = { 0, 0, 0 };
		state.inBaseLayer = 0;
		state.outOffset = { 0, 0, 0 };
		state.outBaseLayer = 0;
		state.extent = attachedRegion->getExtent();
		state.layerCount = attachedRegion->imageSubresource.layerCount;
		state.inMipLevel = attachedRegion->imageSubresource.mipLevel;
		state.outMipLevel = attachedRegion->imageSubresource.mipLevel;

		if (!convertFilter.execute(std::execution::par_unseq,&state))
			os::Printer::log("Something went wrong while converting!", ELL_WARNING);
	}

	return newConvertedImage;
};

//! creates a surface from the file
asset::SAssetBundle CImageLoaderTGA::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	STGAHeader header;
	_file->read(&header, sizeof(STGAHeader));

	core::smart_refctd_ptr<ICPUBuffer> colorMap; // not used, but it's texel buffer may be useful in future
	const auto bytesPerTexel = header.PixelDepth / 8;

	if (header.IdLength) // skip image identification field
		_file->seek(header.IdLength, true);

	if (header.ColorMapType)
	{
		auto colorMapEntryByteSize = header.ColorMapEntrySize / 8 * header.ColorMapLength;
		auto colorMapEntryBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(colorMapEntryByteSize);
		_file->read(colorMapEntryBuffer->getPointer(), header.ColorMapEntrySize / 8 * header.ColorMapLength);
		
		switch ( header.ColorMapEntrySize ) // convert to 32-bit color map since input is dependend to header.ColorMapEntrySize, so it may be 8, 16, 24 or 32 bits per entity
		{
			case STB_8_BITS:
				colorMap = createSingleRowBufferFromRawData<EF_R8_SRGB, EF_R8G8B8A8_SRGB>(colorMapEntryBuffer);
				break;
			case STB_16_BITS:
				colorMap = createSingleRowBufferFromRawData<EF_A1R5G5B5_UNORM_PACK16, EF_R8G8B8A8_SRGB>(colorMapEntryBuffer);
				break;
			case STB_24_BITS:
				colorMap = createSingleRowBufferFromRawData<EF_B8G8R8_SRGB, EF_R8G8B8A8_SRGB>(colorMapEntryBuffer);
				break;
			case STB_32_BITS:
				colorMap = createSingleRowBufferFromRawData<EF_B8G8R8A8_SRGB, EF_R8G8B8A8_SRGB>(colorMapEntryBuffer);
				break;
		}
	}

	ICPUImage::SCreationParams imgInfo;
	imgInfo.type = ICPUImage::ET_2D;
	imgInfo.extent.width = header.ImageWidth;
	imgInfo.extent.height = header.ImageHeight;
	imgInfo.extent.depth = 1u;
	imgInfo.mipLevels = 1u;
	imgInfo.arrayLayers = 1u;
	imgInfo.samples = ICPUImage::ESCF_1_BIT;
	imgInfo.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);

	auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
	core::smart_refctd_ptr<ICPUBuffer> texelBuffer = nullptr;

	ICPUImage::SBufferCopy& region = regions->front();

	region.imageSubresource.mipLevel = 0u;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.bufferOffset = 0u;
	region.bufferImageHeight = 0u;
	region.imageOffset = { 0u, 0u, 0u };
	region.imageExtent = imgInfo.extent;

	// read image
	size_t endBufferSize = {};
	
	switch (header.ImageType)
	{
		case STIT_NONE:
		{
			os::Printer::log("The given TGA doesn't have image data", _file->getFileName().c_str(), ELL_ERROR);
			return {};
		}
		case STIT_UNCOMPRESSED_RGB_IMAGE: [[fallthrough]];
		case STIT_UNCOMPRESSED_GRAYSCALE_IMAGE: [[fallthrough]];
		{
			region.bufferRowLength = calcPitchInBlocks(region.imageExtent.width, getTexelOrBlockBytesize(EF_R8G8B8_SRGB));
			const int32_t imageSize = endBufferSize = region.imageExtent.height * region.bufferRowLength * bytesPerTexel;
			texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(imageSize);
			_file->read(texelBuffer->getPointer(), imageSize);
		}
		break;
		case STIT_RLE_TRUE_COLOR_IMAGE: 
		{
			region.bufferRowLength = calcPitchInBlocks(region.imageExtent.width, bytesPerTexel);
			const auto bufferSize = endBufferSize = region.imageExtent.height * region.bufferRowLength * bytesPerTexel;
			loadCompressedImage(_file, header, bufferSize, texelBuffer);
			break;
		}
		default:
		{
			os::Printer::log("Unsupported TGA file type", _file->getFileName().c_str(), ELL_ERROR);
            return {};
		}
	}

	bool flip = (header.ImageDescriptor & 0x20) == 0;

	core::smart_refctd_ptr<ICPUImage> newConvertedImage;

	switch(header.PixelDepth)
	{
		case 8:
			{
				if (header.ImageType != 3)
				{
					os::Printer::log("Loading 8-bit non-grayscale is NOT supported.", ELL_ERROR);		
                    return {};
				}

				newConvertedImage = createAndconvertImageData<asset::EF_R8_SRGB, asset::EF_R8_SRGB>(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		case 16:
			{
				newConvertedImage = createAndconvertImageData<asset::EF_A1R5G5B5_UNORM_PACK16, asset::EF_A1R5G5B5_UNORM_PACK16>(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		case 24:
			{
				newConvertedImage = createAndconvertImageData<asset::EF_R8G8B8_SRGB, asset::EF_R8G8B8_SRGB>(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		case 32:
			{
			newConvertedImage = createAndconvertImageData<asset::EF_R8G8B8A8_SRGB, asset::EF_R8G8B8A8_SRGB>(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		default:
			os::Printer::log("Unsupported TGA format", _file->getFileName().c_str(), ELL_ERROR);
			break;
	}

	core::smart_refctd_ptr<ICPUImage> image = newConvertedImage;
	if (newConvertedImage->getCreationParameters().format == asset::EF_R8_SRGB)
		image = asset::IImageAssetHandlerBase::convertR8ToR8G8B8Image(newConvertedImage);

	if (!image)
		return {};

    return SAssetBundle(nullptr,{std::move(image)});
}

} // end namespace video
} // end namespace nbl

#endif