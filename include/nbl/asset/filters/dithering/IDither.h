// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_DITHER_H_INCLUDED__
#define __NBL_ASSET_I_DITHER_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"

namespace nbl
{
namespace asset
{
//! Abstract Data Type for CDither class
/*
            Holds top level state for dithering and
            provides some methods for proper another
            base CRTP class implementation - CDither
        */

class IDither
{
public:
    virtual ~IDither() {}

    //! Base state interface class
    /*
                    Holds texel range of an image
                */

    class IState
    {
    public:
        virtual ~IState() {}

        struct TexelRange
        {
            VkOffset3D offset = {0u, 0u, 0u};
            VkExtent3D extent = {0u, 0u, 0u};
        };
    };

    virtual float pGet(const IState* state, const core::vectorSIMDu32& pixelCoord, const int32_t& channel) = 0;
};
}
}

#endif