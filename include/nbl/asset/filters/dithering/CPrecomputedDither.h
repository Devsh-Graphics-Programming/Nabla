// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_PRECOMPUTED_DITHER_H_INCLUDED__
#define __NBL_ASSET_C_PRECOMPUTED_DITHER_H_INCLUDED__

#include "../include/nbl/asset/filters/dithering/CDither.h"
#include "../include/nbl/asset/filters/CFlattenRegionsImageFilter.h"

namespace nbl
{
	namespace asset
	{
		//! A class to apply dithering to an image using precomputed dithering image
		/*
			
		*/

		class CPrecomputedDither : public CDither<CPrecomputedDither>
		{
			public:
				CPrecomputedDither() {}
				virtual ~CPrecomputedDither() {}

				//! State of precomputed dithering class
				/*
					The state requires only input dithering image
					view which image's buffer will be used in dithering 
					process in extent of given mipmap as
					subresourceRange.baseMipLevel.
				*/
				
				class CState : public CDither::CState
				{
					public:
						CState(const asset::ICPUImageView* const ditheringImageView) 
						{
							const bool isBC = asset::isBlockCompressionFormat(ditheringImageView->getCreationParameters().format);
							assert(!isBC); // TODO: log "Precomputed dither image musn't be a BC format!"

							const bool isCorrectChannelCount = asset::getFormatChannelCount(ditheringImageView->getCreationParameters().format) == 4;
							assert(isCorrectChannelCount); // TODO: log "Precomputed dither image must contain all the rgba channels!"

							using FLATTEN_FILTER = CFlattenRegionsImageFilter;
							FLATTEN_FILTER flattenFilter;
							FLATTEN_FILTER::state_type state;

							state.inImage = ditheringImageView->getCreationParameters().image.get();
							bool status = flattenFilter.execute(&state);
							assert(status);
							flattenDitheringImage = std::move(state.outImage);

							const uint32_t& chosenMipmap = ditheringImageView->getCreationParameters().subresourceRange.baseMipLevel;

							const auto& creationParams = flattenDitheringImage->getCreationParameters();
							const auto& extent = flattenDitheringImage->getMipSize(chosenMipmap);
							const size_t newDecodeBufferSize = extent.x * extent.y * extent.z * creationParams.arrayLayers * decodeTexelByteSize;
								
							const hlsl::uint32_t3 decodeBufferByteStrides = TexelBlockInfo(decodeFormat).convert3DTexelStridesTo1DByteStrides(hlsl::uint32_t3(extent.x, extent.y, extent.z));
							auto decodeFlattenBuffer = ICPUBuffer::create({ newDecodeBufferSize });
							decodeFlattenBuffer->setContentHash(IPreHashed::INVALID_HASH);
					
							auto* inData = reinterpret_cast<uint8_t*>(flattenDitheringImage->getBuffer()->getPointer());
							auto* outData = reinterpret_cast<uint8_t*>(decodeFlattenBuffer->getPointer());

							auto decode = [&](uint32_t readBlockArrayOffset, hlsl::uint32_t4 readBlockPos) -> void
							{
								const hlsl::uint32_t4& localOutPos = readBlockPos;

								auto* inDataAdress = inData + readBlockArrayOffset;
								const void* inSourcePixels[] = { inDataAdress, nullptr, nullptr, nullptr };

								double decodeBuffer[forcedChannels] = {};
								
								asset::decodePixelsRuntime(creationParams.format, inSourcePixels, decodeBuffer, 0, 0);
								const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(localOutPos, decodeBufferByteStrides);
								asset::encodePixels<decodeFormat>(outData + offset, decodeBuffer);
							};

							CBasicImageFilterCommon::executePerRegion(flattenDitheringImage.get(), decode, flattenDitheringImage->getRegions(chosenMipmap));

							auto decodeCreationParams = creationParams;
							decodeCreationParams.format = decodeFormat;
							decodeCreationParams.mipLevels = 1;

							auto decodeFlattenImage = ICPUImage::create(std::move(decodeCreationParams));
							auto decodeFlattenRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1);
							*decodeFlattenRegions->begin() = *flattenDitheringImage->getRegions().data();
							decodeFlattenRegions->begin()->imageSubresource.baseArrayLayer = 0;

							decodeFlattenImage->setBufferAndRegions(std::move(decodeFlattenBuffer), decodeFlattenRegions);
							flattenDitheringImage = std::move(decodeFlattenImage);
							{
								ditherImageData.buffer = flattenDitheringImage->getBuffer();
								ditherImageData.format = decodeFormat;
								ditherImageData.strides = decodeBufferByteStrides;
								texelRange.extent = { extent.x, extent.y, extent.z };
							}
						}

						virtual ~CState() {}

						const auto& getDitherImageData() const { return ditherImageData; }

					private:

						static constexpr auto decodeFormat = EF_R32G32B32A32_SFLOAT;
						static constexpr auto decodeTexelByteSize = asset::getTexelOrBlockBytesize<decodeFormat>();
						static constexpr auto forcedChannels = 4;

						core::smart_refctd_ptr<ICPUImage> flattenDitheringImage;

						struct
						{
							const asset::ICPUBuffer* buffer = nullptr;
							hlsl::uint32_t4 strides;
							asset::E_FORMAT format;
						} ditherImageData;
				};

				using state_type = CState;

				//! Get channel texel value from dithered image
				/*
					@param state Input state
					@param pixelCoord Current pixel coordinate of processing input image
					we will be applying dithering to in \btexels\b!
					@param channel Current channel
				*/

				static float get(const state_type* state, const hlsl::uint32_t4& pixelCoord, const int32_t& channel) 
				{
					const auto& ditherImageData = state->getDitherImageData();
					const hlsl::uint32_t4 tiledPixelCoord(pixelCoord.x % (state->texelRange.extent.width - 1), pixelCoord.y % (state->texelRange.extent.height -1), pixelCoord.z, pixelCoord.w);
					const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(tiledPixelCoord, ditherImageData.strides);

					return *(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ditherImageData.buffer->getPointer()) + offset) + channel);
				}
		};
	}
}

#endif