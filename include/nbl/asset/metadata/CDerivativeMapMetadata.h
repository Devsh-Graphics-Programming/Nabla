// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_DERIVATIVE_MAP_METADATA_H_INCLUDED_
#define _NBL_ASSET_C_DERIVATIVE_MAP_METADATA_H_INCLUDED_

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/metadata/IImageViewMetadata.h"

namespace nbl::asset
{
class CDerivativeMapMetadata final : public IAssetMetadata
{
public:
    class CImageView : public IImageViewMetadata
    {
    public:
        CImageView(const float* _scale, bool isotropic)
            : IImageViewMetadata({ECP_PASS_THROUGH, EOTF_IDENTITY})
        {
            scale[0] = _scale[0];
            if(isotropic)
                scale[1] = _scale[0];
            else
                scale[1] = _scale[1];
        }

        inline CImageView& operator=(CImageView&& other)
        {
            IImageViewMetadata::operator=(std::move(other));
            scale[0] = other.scale[0];
            scale[1] = other.scale[1];
            return *this;
        }

        float scale[2];
    };

    CDerivativeMapMetadata(ICPUImageView* imageView, const float* _scale, bool isotropic)
        : IAssetMetadata(), m_metaStorage(_scale, isotropic)
    {
        IAssetMetadata::insertAssetSpecificMetadata(imageView, &m_metaStorage);
    }

    static inline constexpr const char* LoaderName = "CDerivativeMapCreator";
    const char* getLoaderName() const override { return LoaderName; }

private:
    CImageView m_metaStorage;
};

}

#endif