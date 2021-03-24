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

class CPLYMetadata final : public IAssetMetadata
{
    public:
        class CRenderpassIndependentPipeline : public IRenderpassIndependentPipelineMetadata
        {
            public:
                CRenderpassIndependentPipeline() : IRenderpassIndependentPipelineMetadata(), m_hash(0xdeadbeefu) {}
                CRenderpassIndependentPipeline(CRenderpassIndependentPipeline&& other)
                {
                    operator=(std::move(other));
                }

                CRenderpassIndependentPipeline& operator=(const CRenderpassIndependentPipeline&) = delete;
                inline CRenderpassIndependentPipeline& operator=(CRenderpassIndependentPipeline&& other)
                {
                    IRenderpassIndependentPipelineMetadata::operator=(std::move(other));
                    std::swap(m_hash,other.m_hash);
                    return *this;
                }

                uint32_t m_hash;
        };
            
        template<class InContainer>
        CPLYMetadata(InContainer&& hashContainer, core::smart_refctd_dynamic_array<IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>&& _semanticStorage) : 
            IAssetMetadata(), m_metaStorage(createContainer<CRenderpassIndependentPipeline>(hashContainer.size())), m_semanticStorage(std::move(_semanticStorage))
        {
            auto metaIt = m_metaStorage->begin();
            for (auto hash : hashContainer)
                (metaIt++)->m_hash = hash;
        }

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