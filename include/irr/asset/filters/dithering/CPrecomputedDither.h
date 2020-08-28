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
					in full extent of image.
				*/
				
				class CState : public CDither::CState
				{
					public:
						CState(const asset::ICPUImage* const ditheringImage) 
						{
							using FLATTEN_FILTER = CFlattenRegionsImageFilter;
							FLATTEN_FILTER flattenFilter;
							FLATTEN_FILTER::state_type state;

							state.inImage = const_cast<asset::ICPUImage*>(ditheringImage); // TODO change quls in the filter
							assert(flattenFilter.execute(&state));
							flattenDitheringImage = std::move(state.outImage);

							ditherImageData.buffer = flattenDitheringImage->getBuffer();
							const auto extent = flattenDitheringImage->getMipSize();
							ditherImageData.format = flattenDitheringImage->getCreationParameters().format;
							ditherImageData.strides = TexelBlockInfo(ditherImageData.format).convert3DTexelStridesTo1DByteStrides(extent);
							texelRange.extent = { extent.x, extent.y, extent.z };

							assert(!asset::isBlockCompressionFormat(ditherImageData.format), "Precomputed dither image musn't be a BC format!");
							assert(asset::getFormatChannelCount(ditherImageData.format) == 4, "Precomputed dither image must contain all the rgba channels!");
						}

						virtual ~CState() {}

						const auto& getDitherImageData() const { return ditherImageData; }

					private:

						core::smart_refctd_ptr<ICPUImage> flattenDitheringImage;

						struct
						{
							const asset::ICPUBuffer* buffer;
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

					const auto* dstPointer = reinterpret_cast<const uint8_t*>(ditherImageData.buffer->getPointer()) + offset;
					const void* srcPointers[] = { dstPointer, nullptr, nullptr, nullptr };

					constexpr uint8_t MAX_CHANNELS = 4;
					double decodeBuffer[MAX_CHANNELS];
					asset::decodePixelsRuntime(ditherImageData.format, srcPointers, decodeBuffer, 0, 0); // little slow

					// TODO - read from DECODED buffer to not wasting time on decodePixelsRuntime and decoding x channels at one execution of get

					return static_cast<float>(decodeBuffer[channel]);
				}
		};
	}
}

#endif // __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__