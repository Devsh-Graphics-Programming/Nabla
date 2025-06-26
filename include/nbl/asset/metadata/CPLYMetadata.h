// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_PLY_METADATA_H_INCLUDED_
#define _NBL_ASSET_C_PLY_METADATA_H_INCLUDED_


#include "nbl/asset/metadata/IAssetMetadata.h"


namespace nbl::asset
{

class CPLYMetadata final : public IAssetMetadata
{
    public:
#if 0
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
#endif           
        CPLYMetadata() : IAssetMetadata() {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CPLYMeshFileLoader";
        const char* getLoaderName() const override { return LoaderName; }
};

}
#endif