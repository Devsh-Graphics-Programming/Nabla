// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CImageLoaderTGA.h"

#ifdef _NBL_COMPILE_WITH_TGA_LOADER_

#include "nbl/system/IFile.h"

#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/ICPUImage.h"

#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

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
void CImageLoaderTGA::loadCompressedImage(system::IFile *file, const STGAHeader& header, const uint32_t wholeSizeWithPitchInBytes, core::smart_refctd_ptr<ICPUBuffer>& bufferData) const
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
		{
			system::IFile::success_t success;
			file->read(success, &chunkheader, 0, sizeof(uint8_t)); // Read The Chunk's Header
			if (!success)
				return; // TODO: log error
		}
		if(chunkheader < 128) // If The Chunk Is A 'RAW' Chunk
		{
			chunkheader++; // Add 1 To The Value To Get Total Number Of Raw Pixels

			system::IFile::success_t success;
			file->read(success, &data[currentByte], 0, bytesPerPixel * chunkheader);
			if (!success)
				return; // TODO: log
			currentByte += bytesPerPixel * chunkheader;
		}
		else
		{
			// thnx to neojzs for some fixes with this code

			// If It's An RLE Header
			chunkheader -= 127; // Subtract 127 To Get Rid Of The ID Bit

			int32_t dataOffset = currentByte;
			system::IFile::success_t success;
			file->read(success, &data[currentByte], 0, bytesPerPixel);
			if (!success)
				return; // TODO: log
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
bool CImageLoaderTGA::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
	if (!_file)
		return false;
	
	STGAFooter footer;
	memset(&footer, 0, sizeof(STGAFooter));
	{
		system::IFile::success_t success;
		_file->read(success, &footer, _file->getSize() - sizeof(STGAFooter), sizeof(STGAFooter));
		if (!success)
			return false;
	}
	// 16 bytes for "TRUEVISION-XFILE", 17th byte is '.', and the 18th byte contains '\0'.
	if (strncmp(footer.Signature, "TRUEVISION-XFILE.", 18u) != 0)
	{
		logger.log("Invalid (non-TGA) file!", system::ILogger::ELL_ERROR);
		return false;
	}

	float gamma = 0.f;

	if (footer.ExtensionOffset == 0)
	{
		logger.log("Gamma information is not present! Assuming 2.333333", system::ILogger::ELL_WARNING);
		gamma = 2.333333f;
	}
	else
	{
		STGAExtensionArea extension;
		system::IFile::success_t success;
		_file->read(success, &extension, footer.ExtensionOffset, sizeof(STGAExtensionArea));
		if (success)
			gamma = extension.Gamma;
		
		if (gamma == 0.0f)
		{
			logger.log("Gamma information is not present! Assuming 2.233333", system::ILogger::ELL_WARNING);
			gamma = 2.233333f;
		}
		
		// TODO - pass gamma to LoadAsset()?
		// Actually I think metadata will be in used here in near future
	}
	
	
	return true;
}

/*
	Targa formats needs two y-axis flips. The first is a flip to get the Y conforms to OpenGL coords.
	The second flip is defined from within the .tga file itself (header.ImageDescriptor & 0x20).
	if flip - perform two flips (OpenGL + Targa) = no flipping. Don't flip the image at all in that case
	if not - do an OpenGL flip
*/

core::smart_refctd_ptr<ICPUImage> createImage(ICPUImage::SCreationParams& imgInfo, core::smart_refctd_ptr<ICPUBuffer>&& texelBuffer, bool flip)
{
	core::smart_refctd_ptr<ICPUImage> inputCreationImage;
	{
		auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
		ICPUImage::SBufferCopy& region = regions->front();
		
		region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.bufferOffset = 0u;
		region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(imgInfo.extent.width, asset::getTexelOrBlockBytesize(imgInfo.format));
		region.bufferImageHeight = 0u;
		region.imageOffset = { 0u, 0u, 0u };
		region.imageExtent = imgInfo.extent;

		inputCreationImage = asset::ICPUImage::create(std::move(imgInfo));
		inputCreationImage->setBufferAndRegions(std::move(texelBuffer), regions);
		
		bool OpenGlFlip = flip;
		if (OpenGlFlip)
			asset::IImageAssetHandlerBase::performImageFlip(inputCreationImage);
	}

	return inputCreationImage;
};

//! creates a surface from the file
asset::SAssetBundle CImageLoaderTGA::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	STGAHeader header;
	{
		system::IFile::success_t success;
		_file->read(success,&header,0,sizeof(header));
		if (!success)
			return {};
	}

	const auto bytesPerTexel = header.PixelDepth / 8;

	size_t offset = sizeof header;
	if (header.IdLength) // skip image identification field
		offset += header.IdLength;

	if (header.ColorMapType)
	{
		auto colorMapEntryByteSize = header.ColorMapEntrySize / 8 * header.ColorMapLength;
		auto colorMapEntryBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(colorMapEntryByteSize);
		{
			system::IFile::success_t success;
			_file->read(success, colorMapEntryBuffer->getPointer(), offset, header.ColorMapEntrySize / 8 * header.ColorMapLength);
			if (!success)
				return {};
		}
		offset += header.ColorMapEntrySize / 8 * header.ColorMapLength;
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
	
	region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
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
			_params.logger.log("The given TGA doesn't have image data", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
			return {};
		}
		case STIT_UNCOMPRESSED_RGB_IMAGE: [[fallthrough]];
		case STIT_UNCOMPRESSED_GRAYSCALE_IMAGE:
		{
			region.bufferRowLength = calcPitchInBlocks(region.imageExtent.width, getTexelOrBlockBytesize(EF_R8G8B8_SRGB));
			const int32_t imageSize = endBufferSize = region.imageExtent.height * region.bufferRowLength * bytesPerTexel;
			texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(imageSize);
			{
				system::IFile::success_t success;
				_file->read(success, texelBuffer->getPointer(), offset, imageSize);
				if (!success)
					return {};
			}
			offset += imageSize;
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
			_params.logger.log("Unsupported TGA file type", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
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
					_params.logger.log("Loading 8-bit non-grayscale is NOT supported.", system::ILogger::ELL_ERROR);
					return {};
				}
				
				imgInfo.format = asset::EF_R8_SRGB;
				newConvertedImage = createImage(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		case 16:
			{
				imgInfo.format = asset::EF_A1R5G5B5_UNORM_PACK16;
				newConvertedImage = createImage(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		case 24:
			{
				imgInfo.format = asset::EF_R8G8B8_SRGB;
				newConvertedImage = createImage(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		case 32:
			{
				imgInfo.format = asset::EF_R8G8B8A8_SRGB;
				newConvertedImage = createImage(imgInfo, std::move(texelBuffer), flip);
			}
			break;
		default:
			_params.logger.log("Unsupported TGA format %s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
			break;
	}

	core::smart_refctd_ptr<ICPUImage> image = newConvertedImage;

	if (!image)
		return {};

    return SAssetBundle(nullptr,{std::move(image)});
}

} // end namespace video
} // end namespace nbl

#endif