// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_OPENEXR_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_OPENEXR_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl 
{
namespace asset
{

class COpenEXRMetadata final : public IAssetMetadata
{
    public:
        class CImage : public IImageMetadata
        {
            public:
                CImage(const ColorSemantic& _colorSemantic) : IImageMetadata(_colorSemantic) {}

                inline CImage& operator=(CImage&& other)
                {
                    IImageMetadata::operator=(std::move(other));
                    std::swap(name,other.name);
                    return *this;
                }

                std::string name;
        };
        using meta_container_t = core::refctd_dynamic_array<CImage>;

        COpenEXRMetadata(uint32_t imageCount) : IAssetMetadata(), m_metaStorage(meta_container_t::create_dynamic_array(imageCount),core::dont_grab)
        {
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CImageLoaderOpenEXR";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        using meta_container_t = core::refctd_dynamic_array<CImage>;
        core::smart_refctd_ptr<meta_container_t> m_metaStorage;

        friend class CImageLoaderOpenEXR;
        inline void addMeta(uint32_t offset, const ICPUImage* image, const IImageMetadata::ColorSemantic& _colorSemantic, std::string&& _name)
        {
            auto& meta = m_metaStorage->operator[](offset);
            meta = CImage(_colorSemantic);
            meta.name = std::move(_name);

            IAssetMetadata::insertAssetSpecificMetadata(image,&meta);
        }
};

}   
}

#endif