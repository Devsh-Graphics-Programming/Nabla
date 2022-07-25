// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_PLY_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_PLY_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl 
{
namespace asset
{

class NBL_API CPLYMetadata final : public IAssetMetadata
{
    public:
        class CRenderpassIndependentPipeline : public IRenderpassIndependentPipelineMetadata
        {
            public:
                CRenderpassIndependentPipeline(CRenderpassIndependentPipeline&& _other) : CRenderpassIndependentPipeline()
                {
                    CRenderpassIndependentPipeline::operator=(std::move(_other));
                }
                template<typename... Args>
                CRenderpassIndependentPipeline(Args&&... args) : IRenderpassIndependentPipelineMetadata(std::forward<Args>(args)...) {}

                inline CRenderpassIndependentPipeline& operator=(CRenderpassIndependentPipeline&& other)
                {
                    IRenderpassIndependentPipelineMetadata::operator=(std::move(other));
                    return *this;
                }
        };
           
        CPLYMetadata(uint32_t pplnCount, core::smart_refctd_dynamic_array<IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>&& _semanticStorage) :
            IAssetMetadata(), m_metaStorage(createContainer<CRenderpassIndependentPipeline>(pplnCount)), m_semanticStorage(std::move(_semanticStorage)) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CPLYMeshFileLoader";
        const char* getLoaderName() const override { return LoaderName; }

        //!
		inline const CRenderpassIndependentPipeline* getAssetSpecificMetadata(const ICPURenderpassIndependentPipeline* asset) const
		{
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
			return static_cast<const CRenderpassIndependentPipeline*>(found);
		}

    private:
        meta_container_t<CRenderpassIndependentPipeline> m_metaStorage;
        core::smart_refctd_dynamic_array<IRenderpassIndependentPipelineMetadata::ShaderInputSemantic> m_semanticStorage;

        friend class CPLYMeshFileLoader;
        inline void placeMeta(uint32_t offset, const ICPURenderpassIndependentPipeline* ppln)
        {
            auto& meta = m_metaStorage->operator[](offset);
            meta.m_inputSemantics = {m_semanticStorage->begin(),m_semanticStorage->end()};
            IAssetMetadata::insertAssetSpecificMetadata(ppln,&meta);
        }
};

}
}

#endif