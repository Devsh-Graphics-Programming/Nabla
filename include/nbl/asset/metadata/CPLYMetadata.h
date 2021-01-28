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

class CPLYPipelineMetadata final : public IRenderpassIndependentPipelineMetadata
{
    public:
            
        CPLYPipelineMetadata(uint32_t _hash, core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
            : m_hash(_hash), m_shaderInputs(std::move(_inputs)) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CPLYMeshFileLoader";
        const char* getLoaderName() const override { return LoaderName; }

        uint32_t getHashVal() const { return m_hash; }

    private:
        uint32_t m_hash;
        core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
};

}
}

#endif