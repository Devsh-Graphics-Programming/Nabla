#ifndef __IRR_C_PLY_PIPELINE_METADATA_H_INCLUDED__
#define __IRR_C_PLY_PIPELINE_METADATA_H_INCLUDED__

#include "irr/asset/IPipelineMetadata.h"

namespace irr 
{
    namespace asset
    {
        class CPLYPipelineMetadata final : public IPipelineMetadata
        {
        public:
            
            CPLYPipelineMetadata(uint32_t _hash, core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
                : m_hash(_hash), m_shaderInputs(std::move(_inputs)) {}

            core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override { return { m_shaderInputs->begin(), m_shaderInputs->end() }; }
            const char* getLoaderName() const override { return "CPLYMeshFileLoader"; }

            uint32_t getHashVal() const { return m_hash; }

        private:
            uint32_t m_hash;
            core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
        };

    }
}

#endif // __IRR_C_PLY_PIPELINE_METADATA_H_INCLUDED__