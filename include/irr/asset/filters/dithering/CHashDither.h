// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_HASH_DITHER_H_INCLUDED__
#define __IRR_C_HASH_DITHER_H_INCLUDED__

#include "../include/irr/asset/filters/dithering/CDither.h"

namespace irr
{
	namespace asset
	{
		class CHashDither : public CDither<CHashDither>
		{
			public:
				CHashDither() {}
				virtual ~CHashDither() {}

				class CState
				{
					public:
						CState() {}
						virtual ~CState() {}
				};

				using state_type = CState;

				static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord)
				{
					// TODO: to define in future
				}
		};
	}
}

#endif // __IRR_C_HASH_DITHER_H_INCLUDED__