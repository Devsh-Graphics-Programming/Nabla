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
                using IImageMetadata::IImageMetadata;

                inline CImage& operator=(CImage&& other)
                {
                    IImageMetadata::operator=(std::move(other));
                    std::swap(m_name,other.m_name);
                    return *this;
                }

                std::string m_name;
        };

        COpenEXRMetadata(uint32_t imageCount) : IAssetMetadata(), m_metaStorage(createContainer<CImage>(imageCount))
        {
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CImageLoaderOpenEXR";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        meta_container_smart_ptr_t<CImage> m_metaStorage;

        friend class CImageLoaderOpenEXR;
        template<typename... Args>
        inline void addMeta(uint32_t offset, const ICPUImage* image, std::string&& _name, Args&&... args)
        {
            auto& meta = m_metaStorage->operator[](offset);
            meta = CImage(std::forward<Args>(args)...);
            meta.m_name = std::move(_name);

            IAssetMetadata::insertAssetSpecificMetadata(image,&meta);
        }
};

}   
}

#endif