// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_WHITE_NOISE_DITHER_H_INCLUDED__
#define __NBL_ASSET_C_WHITE_NOISE_DITHER_H_INCLUDED__

#include "../include/nbl/asset/filters/dithering/CDither.h"

namespace nbl
{
	namespace asset
	{
		//! A class to apply dithering to an image using wang hash function
		/*
			The wang hash function has quite similar distribution to white noise.
		*/

		class NBL_API CWhiteNoiseDither : public CDither<CWhiteNoiseDither>
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
						using REQUIRED_TYPE = uint32_t;
						REQUIRED_TYPE seed = ((channel * uint8_t(~0) + pixelCoord.z) * uint8_t(~0) + pixelCoord.y) * uint8_t(~0) + pixelCoord.x;

						seed = (seed ^ 61) ^ (seed >> 16);
						seed *= 9;
						seed = seed ^ (seed >> 4);
						seed *= 0x27d4eb2d;
						seed = seed ^ (seed >> 15);
						return seed;
					};
					
					const auto hash = static_cast<float>(getWangHash());
					return hash / float(~0u);
				}
		};
	}
}

#endif