// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_STL_PIPELINE_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_STL_PIPELINE_METADATA_H_INCLUDED__

#include "nbl/asset/IPipelineMetadata.h"

namespace nbl 
{
    namespace asset
    {
        class CSTLPipelineMetadata final : public IPipelineMetadata
        {
        public:
            
            CSTLPipelineMetadata(core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
                : m_shaderInputs(std::move(_inputs)) {}

            core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override { return { m_shaderInputs->begin(), m_shaderInputs->end() }; }

            _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CSTLMeshFileLoader";
            const char* getLoaderName() const override { return LoaderName; }

        private:
            core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
        };

    }
}

#endif