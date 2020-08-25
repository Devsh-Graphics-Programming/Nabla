// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__
#define __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"

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
					which will be completly used in dithering process
					in full extent of image.
				*/
				
				class CState : public CDither::CState
				{
					public:
						CState(core::smart_refctd_ptr<asset::ICPUImage> _ditheringImage) 
							: ditheringImage(core::smart_refctd_ptr<asset::ICPUImage>(_ditheringImage))
						{
							ditherImageData.buffer = ditheringImage->getBuffer();
							const auto extent = ditheringImage->getMipSize();
							ditherImageData.format = ditheringImage->getCreationParameters().format;
							ditherImageData.strides = TexelBlockInfo(ditherImageData.format).convert3DTexelStridesTo1DByteStrides(extent);
							texelRange = { extent.x, extent.y, extent.z };

							assert(!asset::isBlockCompressionFormat(ditherImageData.format), "Precomputed dither image musn't be a BC format!");
							assert(asset::getFormatChannelCount(ditherImageData.format) == 4, "Precomputed dither image must contain all the rgba channels!");
						}

						virtual ~CState() {}

						const auto& getDitherImageData() const { return ditherImageData; }

					private:

						struct
						{
							asset::ICPUBuffer* buffer;
							core::vectorSIMDu32 strides;
							asset::E_FORMAT format;
						} ditherImageData;

						core::smart_refctd_ptr<asset::ICPUImage> ditheringImage;
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
					const core::vectorSIMDu32 tiledPixelCoord(pixelCoord.x % state->texelRange.extent.width - 1, pixelCoord.y % state->texelRange.extent.height -1, pixelCoord.z);
					const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(tiledPixelCoord, ditherImageData.strides);

					const auto* dstPointer = reinterpret_cast<const uint8_t*>(ditherImageData.buffer->getPointer()) + offset;
					const void* srcPointers[] = { dstPointer, nullptr, nullptr, nullptr };

					constexpr uint8_t MAX_CHANNELS = 4;
					double decodeBuffer[MAX_CHANNELS];
					asset::decodePixelsRuntime(ditherImageData.format, srcPointers, decodeBuffer, 0, 0); // little slow

					return static_cast<float>(decodeBuffer[channel]);
				}
		};
	}
}

#endif // __IRR_C_PRECOMPUTED_DITHER_H_INCLUDED__