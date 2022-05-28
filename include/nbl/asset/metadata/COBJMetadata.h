// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_OBJ_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_OBJ_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/metadata/CMTLMetadata.h"

namespace nbl
{
namespace asset
{

class NBL_API COBJMetadata final : public IAssetMetadata
{
    public:
        using CRenderpassIndependentPipeline = typename CMTLMetadata::CRenderpassIndependentPipeline;
        COBJMetadata(uint32_t pplnCount) : IAssetMetadata(), m_metaStorage(createContainer<CRenderpassIndependentPipeline>(pplnCount))
        {
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CGraphicsPipelineLoaderMTL";
        const char* getLoaderName() const override { return LoaderName; }

        //!
		inline const CRenderpassIndependentPipeline* getAssetSpecificMetadata(const ICPURenderpassIndependentPipeline* asset) const
		{
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
			return static_cast<const CRenderpassIndependentPipeline*>(found);
		}

    private:
        meta_container_t<CRenderpassIndependentPipeline> m_metaStorage;

        friend class COBJMeshFileLoader;
        inline void placeMeta(uint32_t offset, const ICPURenderpassIndependentPipeline* ppln, const CRenderpassIndependentPipeline& _meta)
        {
            auto& meta = m_metaStorage->operator[](offset);
            meta.m_inputSemantics = _meta.m_inputSemantics;
            meta.m_descriptorSet3 = _meta.m_descriptorSet3;
            meta.m_materialParams = _meta.m_materialParams;
            meta.m_name = _meta.m_name;
            meta.m_hash = _meta.m_hash;

            IAssetMetadata::insertAssetSpecificMetadata(ppln,&meta);
        }
};

}
}

#endif
