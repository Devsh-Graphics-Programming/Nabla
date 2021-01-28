// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_STL_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_STL_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl 
{
namespace asset
{

class CSTLPipelineMetadata final : public IRenderpassIndependentPipelineMetadata
{
    public:
            
        CSTLPipelineMetadata(core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
            : m_shaderInputs(std::move(_inputs)) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CSTLMeshFileLoader";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
 };

}
}

#endif