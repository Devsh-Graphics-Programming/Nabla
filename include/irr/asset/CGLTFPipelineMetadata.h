#ifndef __IRR_C_GLTF_PIPELINE_METADATA_H_INCLUDED__
#define __IRR_C_GLTF_PIPELINE_METADATA_H_INCLUDED__

#include "irr/asset/IPipelineMetadata.h"
#include "irr/asset/ICPUDescriptorSet.h"
#include "irr/asset/ICPUPipelineLayout.h"

namespace irr
{
    namespace asset
    {
        class CGLTFPipelineMetadata final : public IPipelineMetadata
        {
        public:
            CGLTFPipelineMetadata(std::string&& _name, core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs) 
                : m_name(std::move(_name)), m_shaderInputs(std::move(_inputs)) {}

            const std::string getMaterialName() const { return m_name; }

            core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override { return { m_shaderInputs->begin(), m_shaderInputs->end() }; }

            _IRR_STATIC_INLINE_CONSTEXPR const char* loaderName = "CGLTFLoader";
            const char* getLoaderName() const override { return loaderName; }

        private:
            std::string m_name;
            core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
        };
    }
}

#endif // __IRR_C_GLTF_PIPELINE_METADATA_H_INCLUDED__
