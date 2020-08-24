// Copyright (C) 2020 - AnastaZIuk
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_DITHER_H_INCLUDED__
#define __IRR_I_DITHER_H_INCLUDED__

#include "irr/core/core.h"
#include "irr/asset/ICPUImage.h"
#include "irr/asset/ICPUImageView.h"

namespace irr
{
    namespace asset
    {
        class IDither
        {
            public:
                virtual ~IDither() {}
                virtual float pGet(const core::vectorSIMDu32& pixelCoord) = 0;

                class IState
                {
                    virtual ~IState() {}

                    struct ImageAsset
                    {
                        core::smart_refctd_ptr<asset::IAsset> asset;
                        asset::IAsset::E_TYPE type;
                    };
                };
        };
    }
}

#endif // __IRR_I_DITHER_H_INCLUDED__