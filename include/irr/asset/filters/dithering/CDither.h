// Copyright (C) 2020 - AnastaZIuk
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_DITHER_H_INCLUDED__
#define __IRR_C_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/IDither.h"

namespace irr
{
	namespace asset
	{
		template<class CRTP>
		class CDither : public IDither
		{
			public:
				CDither() {}
				virtual ~CDither() {}
				
				float pGet(const core::vectorSIMDu32& pixelCoord) override
				{
					return CRTP::get(pixelCoord);
				}

				class CState : public IDither::IState
				{
					public:
						CState() {}
						virtual ~CState() {}

						ImageAsset imageAsset; // do we want to only handle ICPUImage or should I distinguish it to ICPUImageView?
				};

				using state_type = CState;

			private:
			
				float get(const state_type* state, const core::vectorSIMDu32& pixelCoord)
				{
					// use blue noise tile ability to fetch suitable texel pointer from texture
				}
		};
	}
}

#endif // __IRR_C_DITHER_H_INCLUDED__