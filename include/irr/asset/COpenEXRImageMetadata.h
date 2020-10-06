// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__

#include "irr/asset/IImageMetadata.h"

namespace irr 
{
namespace asset
{
    class COpenEXRImageMetadata final : public IImageMetadata
    {
        public:

            COpenEXRImageMetadata(std::string _name, const ColorSemantic& _colorSemantic) : name(_name), IImageMetadata(colorSemantic) {}

            std::string getName() const
            {
                return name;
            }

            _IRR_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CImageLoaderOpenEXR";
            const char* getLoaderName() const override { return LoaderName; }

        private:
            std::string name;
    };

}   
}

#endif // __IRR_C_OPENEXR_IMAGE_METADATA_H_INCLUDED__