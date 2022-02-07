// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_DITHER_H_INCLUDED__
#define __NBL_ASSET_C_DITHER_H_INCLUDED__

#include "nbl/asset/filters/dithering/IDither.h"

namespace nbl
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
        const auto& return_value = static_cast<CRTP*>(this)->get(static_cast<const typename CRTP::CState*>(state), pixelCoord, channel);

#ifdef _NBL_DEBUG
        bool status = return_value >= 0 && return_value <= 1;
        assert(status);
#endif  // _NBL_DEBUG

        return return_value;
    }
};

/*
			Identity Dither is used for not providing any Dither in a state.
		*/

class IdentityDither : public CDither<IdentityDither>
{
public:
    IdentityDither() {}
    virtual ~IdentityDither() {}

    static float get(const state_type* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel)
    {
        return {};
    }
};
}
}

#endif