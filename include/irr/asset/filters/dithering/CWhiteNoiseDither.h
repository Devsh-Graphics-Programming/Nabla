// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_WHITE_NOISE_DITHER_H_INCLUDED__
#define __IRR_C_WHITE_NOISE_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"

namespace irr
{
	namespace asset
	{
		class CWhiteNoiseDither : public CDither<CWhiteNoiseDither>
		{
			public:
				CWhiteNoiseDither() {}
				virtual ~CWhiteNoiseDither() {}

				class CState
				{
					public:
						CState()
							: sampler(std::chrono::high_resolution_clock::now().time_since_epoch().count())
						{
						
						}
						virtual ~CState() {}

						core::RandomSampler sampler;
				};

				using state_type = CState;

				static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel)
				{
					/*

						TODO - handle it somehow

					auto storeToTexel = [nonPremultBlendSemantic, alphaChannel, &sampler, outFormat](value_type* const sample, void* const dstPix) -> void
					{
						if (nonPremultBlendSemantic && sample[alphaChannel] > FLT_MIN * 1024.0 * 512.0)
						{
							for (auto i = 0; i < MaxChannels; i++)
								if (i != alphaChannel)
									sample[i] /= sample[alphaChannel];
						}
						for (auto i = 0; i < MaxChannels; i++)
						{
							//sample[i] = core::clamp<value_type,value_type>(sample[i],0.0,1.0);
							// @Crisspl replace this with epic quantization (actually it would be good if you cached the max and min values for the 4 channels outside the hot loop
							sample[i] += double(sampler.nextSample()) * (asset::getFormatPrecision<value_type>(outFormat, i, sample[i]) / double(~0u));
							sample[i] = core::clamp<value_type, value_type>(sample[i], asset::getFormatMinValue<value_type>(outFormat, i), asset::getFormatMaxValue<value_type>(outFormat, i));
						}
						asset::encodePixels<value_type>(outFormat, dstPix, sample);
					};

					*/
				}
		};
	}
}

#endif // __IRR_C_WHITE_NOISE_DITHER_H_INCLUDED__