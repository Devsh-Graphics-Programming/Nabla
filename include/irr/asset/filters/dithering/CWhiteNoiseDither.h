// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_WHITE_NOISE_DITHER_H_INCLUDED__
#define __IRR_C_WHITE_NOISE_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"

namespace irr
{
	namespace asset
	{
		//! A class to apply dithering to an image using wang hash function
		/*
			The wang hash function has quite similar distribution to white noise.
		*/

		class CWhiteNoiseDither : public CDither<CWhiteNoiseDither>
		{
			public:
				CWhiteNoiseDither() {}
				virtual ~CWhiteNoiseDither() {}

				class CState : public CDither::CState
				{
					public:
						CState() {}
						virtual ~CState() {}
				};

				using state_type = CState;

				static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel)
				{
					auto getWangHash = [&]()
					{
						IImage::SBufferCopy::getLocalByteOffset(pixelCoord, bufferStridesHash);
						uint32_t seed = IImage::SBufferCopy::getLocalByteOffset(pixelCoord, bufferStridesHash) * channel;

						seed = (seed ^ 61) ^ (seed >> 16);
						seed *= 9;
						seed = seed ^ (seed >> 4);
						seed *= 0x27d4eb2d;
						seed = seed ^ (seed >> 15);
						return seed;
					};
					
					const auto hash = static_cast<double>(getWangHash());
					return static_cast<float>(static_cast<double>(hash) / double(~0u));
				}

			private:

				static inline const core::vector3du32_SIMD bufferStridesHash = TexelBlockInfo(EF_R8G8B8A8_UINT).convert3DTexelStridesTo1DByteStrides(core::vector3du32_SIMD(uint16_t(~0), uint16_t(~0), uint16_t(~0)));
		};
	}
}

#endif // __IRR_C_WHITE_NOISE_DITHER_H_INCLUDED__