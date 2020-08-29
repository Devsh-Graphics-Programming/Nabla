// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__
#define __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"
#include "../include/irr/asset/filters/CFlattenRegionsImageFilter.h"

namespace irr
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
					which buffer will be used in dithering process
					in full extent of image. Dithering will be applied
					using first 0th mipmap and layer either.
				*/
				
				class CState : public CDither::CState
				{
					public:
						CState(const asset::ICPUImage* const ditheringImage) 
						{
							const bool isBC = asset::isBlockCompressionFormat(ditheringImage->getCreationParameters().format);
							assert(!isBC, "Precomputed dither image musn't be a BC format!");

							const bool isCorrectChannelCount = asset::getFormatChannelCount(ditheringImage->getCreationParameters().format) == 4;
							assert(isCorrectChannelCount, "Precomputed dither image must contain all the rgba channels!");

							using FLATTEN_FILTER = CFlattenRegionsImageFilter;
							FLATTEN_FILTER flattenFilter;
							FLATTEN_FILTER::state_type state;

							state.inImage = ditheringImage; 
							bool status = flattenFilter.execute(&state);
							assert(status);
							flattenDitheringImage = std::move(state.outImage);

							const auto& creationParams = flattenDitheringImage->getCreationParameters();
							const auto& extent = creationParams.extent;
							const size_t newDecodeBufferSize = extent.width * extent.height * extent.depth * decodeTexelByteSize;
							const core::vector3du32_SIMD decodeBufferByteStrides = TexelBlockInfo(decodeFormat).convert3DTexelStridesTo1DByteStrides(core::vector3du32_SIMD(extent.width, extent.height, extent.depth));
							auto decodeFlattenBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(newDecodeBufferSize);
					
							auto* inData = reinterpret_cast<uint8_t*>(flattenDitheringImage->getBuffer()->getPointer());
							auto* outData = reinterpret_cast<uint8_t*>(decodeFlattenBuffer->getPointer());

							auto decode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
							{
								const core::vectorSIMDu32& localOutPos = readBlockPos;

								auto* inDataAdress = inData + readBlockArrayOffset;
								const void* inSourcePixels[] = { inDataAdress, nullptr, nullptr, nullptr };

								double decodeBuffer[forcedChannels] = {};
								
								asset::decodePixelsRuntime(creationParams.format, inSourcePixels, decodeBuffer, 0, 0);
								const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(localOutPos, decodeBufferByteStrides);
								memcpy(outData + offset, decodeBuffer, decodeTexelByteSize);	
							};

							CBasicImageFilterCommon::executePerRegion(flattenDitheringImage.get(), decode, flattenDitheringImage->getRegions().begin(), flattenDitheringImage->getRegions().end());

							auto decodeCreationParams = creationParams;
							decodeCreationParams.format = decodeFormat;
							decodeCreationParams.mipLevels = 1;
							decodeCreationParams.arrayLayers = 1;

							auto decodeFlattenImage = ICPUImage::create(std::move(decodeCreationParams));
							auto decodeFlattenRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1);
							*decodeFlattenRegions->begin() = *flattenDitheringImage->getRegions().begin();
							decodeFlattenRegions->begin()->imageSubresource.baseArrayLayer = 0;
							decodeFlattenRegions->begin()->imageSubresource.layerCount = 1;

							decodeFlattenImage->setBufferAndRegions(std::move(decodeFlattenBuffer), decodeFlattenRegions);
							flattenDitheringImage = std::move(decodeFlattenImage);
							{
								ditherImageData.buffer = flattenDitheringImage->getBuffer();
								ditherImageData.format = decodeFormat;
								ditherImageData.strides = decodeBufferByteStrides;
								texelRange.extent = extent;
							}
						}

						virtual ~CState() {}

						const auto& getDitherImageData() const { return ditherImageData; }

					private:

						static constexpr auto decodeFormat = EF_R64G64B64A64_SFLOAT;
						static constexpr auto decodeTexelByteSize = asset::getTexelOrBlockBytesize<decodeFormat>();
						static constexpr auto forcedChannels = 4;

						core::smart_refctd_ptr<ICPUImage> flattenDitheringImage;

						struct
						{
							const asset::ICPUBuffer* buffer = nullptr;
							core::vectorSIMDu32 strides;
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

				static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel) 
				{
					const auto& ditherImageData = state->getDitherImageData();
					const core::vectorSIMDu32 tiledPixelCoord(pixelCoord.x % (state->texelRange.extent.width - 1), pixelCoord.y % (state->texelRange.extent.height -1), pixelCoord.z);
					const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(tiledPixelCoord, ditherImageData.strides);

					const auto* channelTexelValue = reinterpret_cast<const double*>(reinterpret_cast<const uint8_t*>(ditherImageData.buffer->getPointer()) + offset) + channel;
					return static_cast<float>(*channelTexelValue);
				}
		};
	}
}

#endif // __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__