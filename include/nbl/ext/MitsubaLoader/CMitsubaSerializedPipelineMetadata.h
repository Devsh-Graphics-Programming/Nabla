// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED__

#include "nbl/asset/IPipelineMetadata.h"

namespace irr
{
    namespace ext
    {
        namespace MitsubaLoader
        {
            using namespace asset;

            class CMitsubaSerializedPipelineMetadata final : public IPipelineMetadata
            {
            public:

                CMitsubaSerializedPipelineMetadata(core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs) :
                    m_shaderInputs(std::move(_inputs)) {}


                core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override { return { m_shaderInputs->begin(), m_shaderInputs->end() }; }

                _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CSerializedLoader";
                const char* getLoaderName() const override { return LoaderName; }

            private:
                core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
            };
        }
    }
}

#endif