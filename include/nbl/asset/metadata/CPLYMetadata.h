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
                CRenderpassIndependentPipeline() = default;
                CRenderpassIndependentPipeline(const CRenderpassIndependentPipeline&) = delete;
                CRenderpassIndependentPipeline(CRenderpassIndependentPipeline&& other)
                {
                    operator=(std::move(other));
                }

                template<typename... Args>
                CRenderpassIndependentPipeline(uint32_t _hash, Args&&... args) : IRenderpassIndependentPipelineMetadata(std::forward<Args>(args)...), m_hash(_hash)
                {
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
        CPLYMetadata(InContainer&& inContainer) : IAssetMetadata(), m_metaStorage(createContainer<CRenderpassIndependentPipeline>(inContainer.size()))
        {
            std::move(inContainer.begin(),inContainer.end(),m_metaStorage->begin());
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CPLYMeshFileLoader";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        meta_container_t<CRenderpassIndependentPipeline> m_metaStorage;

        friend class CPLYMeshFileLoader;
        inline void placeMeta(uint32_t offset, const ICPURenderpassIndependentPipeline* ppln)
        {
            auto& meta = m_metaStorage->operator[](offset);
            IAssetMetadata::insertAssetSpecificMetadata(ppln,&meta);
        }
};

}
}

#endif