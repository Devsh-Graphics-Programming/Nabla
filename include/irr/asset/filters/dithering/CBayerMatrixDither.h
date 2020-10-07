// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#if 0 // TODO

#ifndef __IRR_C_BAYER_MATRIX_DITHER_H_INCLUDED__
#define __IRR_C_BAYER_MATRIX_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"

namespace irr
{
	namespace asset
	{
		class CBayerMatrixDither : public IDither
		{
			public:
				CBayerMatrixDither() {}
				virtual ~CBayerMatrixDither() {}

				class CState
				{
					public:
						CState() {}
						virtual ~CState() {}
				};

				using state_type = CState;
			
				static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel)
				{
					// TODO: to define in future
				}
		};
	}
}

#endif // __IRR_C_BAYER_MATRIX_DITHER_H_INCLUDED__

#endif // TODO