// Copyright (C) 2020 - AnastaZIuk
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_DITHER_H_INCLUDED__
#define __IRR_C_DITHER_H_INCLUDED__

#include "../include/irr/asset/IDither.h"

namespace irr
{
	namespace asset
	{
		//! Base CRTP class for dithering classes
		/*
			There are several dithering classes:

			- CWhiteNoiseDither
			- CBayerMatrixDither
			- CPrecomputedDither

			Each of them put some noise on a processed image.
		*/

		template<class CRTP>
		class CDither : public IDither
		{
			public:
				CDither() {}
				virtual ~CDither() {}

				class CState : public IDither::IState
				{
					public:
						CState() {}
						virtual ~CState() {}

						TexelRange texelRange;
				};

				using state_type = CState;

				float pGet(const IDither::IState* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel) final override
				{
					return get(state, pixelCoord, channel);
				}
				
				float get(const IDither::IState* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel)
				{
					return static_cast<CRTP*>(this)->get(static_cast<const typename CRTP::CState*>(state), pixelCoord, channel);
				}
		};
	}
}

#endif // __IRR_C_DITHER_H_INCLUDED__