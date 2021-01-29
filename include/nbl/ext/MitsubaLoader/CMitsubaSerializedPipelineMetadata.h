// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

class CMitsubaSerializedMetadata final : public asset::IAssetMetadata
{
    public:
        class CRenderpassIndependentPipeline : public asset::IRenderpassIndependentPipelineMetadata
        {
            public:
                using IRenderpassIndependentPipelineMetadata::IRenderpassIndependentPipelineMetadata;
        };

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "ext::MitsubaLoader::CSerializedLoader";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        meta_container_t<CRenderpassIndependentPipeline> m_metaStorage;

        friend class CSerializedLoader;
        inline void addMeta(uint32_t offset, const asset::ICPURenderpassIndependentPipeline* ppln)
        {
            static_assert(false);
            //auto& meta = m_metaStorage->operator[](offset);
            //IAssetMetadata::insertAssetSpecificMetadata(ppln, &meta);
        }
};

}
}
}

#endif