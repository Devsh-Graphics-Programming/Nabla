#ifndef __IRR_C_STL_PIPELINE_METADATA_H_INCLUDED__
#define __IRR_C_STL_PIPELINE_METADATA_H_INCLUDED__

#include "irr/asset/IPipelineMetadata.h"

namespace irr 
{
    namespace asset
    {
        class CSTLPipelineMetadata final : public IPipelineMetadata
        {
        public:
            
            CSTLPipelineMetadata(core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
                : m_shaderInputs(std::move(_inputs)) {}

            core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override { return { m_shaderInputs->begin(), m_shaderInputs->end() }; }

            _IRR_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CSTLMeshFileLoader";
            const char* getLoaderName() const override { return LoaderName; }

        private:
            core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
        };

    }
}

#endif // __IRR_C_STL_PIPELINE_METADATA_H_INCLUDED__