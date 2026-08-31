// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/compile_config.h"

#include "CImageLoaderAVIF.h"

#ifdef _NBL_COMPILE_WITH_AVIF_LOADER_

#include "nbl/system/IFile.h"

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/ICPUImageView.h"

#include "nbl/asset/interchange/CImageHasher.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

#include <string>

#include <avif/avif.h>
#include <memory>

namespace nbl
{
	namespace asset
	{

		//! constructor
		CImageLoaderAVIF::CImageLoaderAVIF()
		{
#ifdef _NBL_DEBUG
			setDebugName("CImageLoaderAVIF");
#endif
		}



		//! destructor
		CImageLoaderAVIF::~CImageLoaderAVIF()
		{}

		//! returns true if the file maybe is able to be loaded by this class
		bool CImageLoaderAVIF::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
		{
			if (!_file)
				return false;

			uint8_t buffer[64];
			system::IFile::success_t success;
			_file->read(success, buffer, 0, sizeof(buffer));
			if (!success)
				return false;

			avifROData roData = { buffer, sizeof(buffer) };
			return avifPeekCompatibleFileType(&roData) == AVIF_TRUE;
		}

		//! creates a surface from the file
		asset::SAssetBundle CImageLoaderAVIF::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file || _file->getSize() > 0xffffffffull)
				return {};

			const std::filesystem::path& Filename = _file->getFileName();

            std::size_t fileSize = _file->getSize();
			std::unique_ptr<uint8_t> input{ new(std::nothrow)uint8_t[fileSize] };
			if (!input)
				return {};

			system::IFile::success_t success;
			_file->read(success, input.get(), 0, fileSize);
			if (!success)
				return {};

			avifDecoder* decoder = avifDecoderCreate();
			if (!decoder)
				return {};
			auto exiter = core::makeRAIIExiter([decoder]() { avifDecoderDestroy(decoder); });

			avifResult result = avifDecoderSetIOMemory(decoder, reinterpret_cast<const uint8_t*>(input.get()), fileSize);
			if (result != AVIF_RESULT_OK)
			{
				_params.logger.log("Error during avifDecoderSetIOMemory: %s", system::ILogger::ELL_ERROR, avifResultToString(result));
				return {};
			}

			result = avifDecoderParse(decoder);
			if (result != AVIF_RESULT_OK)
			{
				_params.logger.log("Error during avifDecoderParse: %s", system::ILogger::ELL_ERROR, avifResultToString(result));
				return {};
			}

			result = avifDecoderNextImage(decoder);
			if (result != AVIF_RESULT_OK && result != AVIF_RESULT_WAITING_ON_IO)
			{
				_params.logger.log("Error during avifDecoderNextImage: %s", system::ILogger::ELL_ERROR, avifResultToString(result));
				return {};
			}

			const uint32_t width = decoder->image->width;
			const uint32_t height = decoder->image->height;

			ICPUImage::SCreationParams imgInfo;
			imgInfo.type = ICPUImage::ET_2D;
			imgInfo.extent.width = width;
			imgInfo.extent.height = height;
			imgInfo.extent.depth = 1u;
			imgInfo.mipLevels = 1u;
			imgInfo.arrayLayers = 1u;
			imgInfo.samples = ICPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			imgInfo.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);

			avifRGBImage rgb;
			avifRGBImageSetDefaults(&rgb, decoder->image);
			

			imgInfo.format = EF_R8G8B8A8_SRGB;
			rgb.format = AVIF_RGB_FORMAT_RGBA;
			
			if (decoder->image->depth > 8)
			{
				imgInfo.format = EF_R16G16B16A16_UNORM;
				rgb.depth = 16;
			}
			else
			{
				rgb.depth = 8;
			}

			CImageHasher contentHasher(imgInfo);

			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
			ICPUImage::SBufferCopy& region = regions->front();
			region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			region.imageSubresource.mipLevel = 0u;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = 1u;
			region.bufferOffset = 0u;
			region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(width, getTexelOrBlockBytesize(imgInfo.format));
			region.bufferImageHeight = 0u; //tightly packed
			region.imageOffset = { 0u, 0u, 0u };
			region.imageExtent = imgInfo.extent;

			uint32_t bpp = rgb.depth == 8 ? 4 : 8; // RGBA8 vs RGBA16
			uint32_t rowspan = region.bufferRowLength * bpp;

			auto buffer = asset::ICPUBuffer::create({ rowspan * height });
			if (!buffer)
				return {};

			rgb.pixels = reinterpret_cast<uint8_t*>(buffer->getPointer());
			rgb.rowBytes = rowspan;

			result = avifImageYUVToRGB(decoder->image, &rgb);
			if (result != AVIF_RESULT_OK)
			{
				_params.logger.log("Error during avifImageYUVToRGB: %s", system::ILogger::ELL_ERROR, avifResultToString(result));
				return {};
			}

			uint8_t* pixels = rgb.pixels;
			for (uint32_t y = 0; y < height; ++y)
			{
				contentHasher.partialHash(0, 0, pixels + (y * rowspan), rowspan);
			}
			contentHasher.hashSeq(0, 0);

			core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imgInfo));
			image->setBufferAndRegions(std::move(buffer), regions);

			auto hash = contentHasher.finalizeSeq();
			image->setContentHash(hash);

			return SAssetBundle(nullptr, { image });
		}

	} // end namespace asset
} // end namespace nbl

#endif




