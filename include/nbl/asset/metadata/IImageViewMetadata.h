// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_IMAGE_VIEW_METADATA_H_INCLUDED_
#define _NBL_ASSET_I_IMAGE_VIEW_METADATA_H_INCLUDED_

#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/format/EColorSpace.h"

namespace nbl::asset
{
//!
class IImageViewMetadata : public core::Interface
{
public:
    struct ColorSemantic
    {
        E_COLOR_PRIMARIES colorSpace;
        ELECTRO_OPTICAL_TRANSFER_FUNCTION transferFunction;
    };

    inline IImageViewMetadata()
        : colorSemantic{ECP_COUNT, EOTF_UNKNOWN} {}
    inline IImageViewMetadata(const ColorSemantic& _colorSemantic)
        : colorSemantic(_colorSemantic) {}

    ColorSemantic colorSemantic;

protected:
    virtual ~IImageViewMetadata() = default;

    inline IImageViewMetadata& operator=(IImageViewMetadata&& other)
    {
        std::swap(colorSemantic, other.colorSemantic);
        return *this;
    }
};

}

#endif
