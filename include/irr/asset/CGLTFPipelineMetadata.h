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

            enum E_ALPHA_MODE : uint16_t
            {
                EAM_OPAQUE = core::createBitmask({0}),
                EAM_MASK = core::createBitmask({1}),
                EAM_BLEND = core::createBitmask({2})
            };

            #include "irr/irrpack.h"
            //! This struct is compliant with GLSL's std140 and std430 layouts
            struct alignas(16) SGLTFMaterialParameters
            {
                struct SPBRMetallicRoughness
                {
                    core::vector4df_SIMD baseColorFactor = core::vector4df_SIMD(1.f, 1.f, 1.f, 1.f);
                    float metallicFactor = 1.f;
                    float roughnessFactor = 1.f;
                };

                SPBRMetallicRoughness metallicRoughness;
                core::vector3df_SIMD emissiveFactor = core::vector3df_SIMD(0, 0, 0);
                E_ALPHA_MODE alphaMode = EAM_OPAQUE;
                float alphaCutoff = 0.5f; 

            } PACK_STRUCT;
            #include "irr/irrunpack.h"
            //VS Intellisense shows error here because it think vectorSIMDf is 32 bytes, but it just Intellisense - it'll build anyway
            static_assert(sizeof(SGLTFMaterialParameters) == (sizeof(SGLTFMaterialParameters::SPBRMetallicRoughness::baseColorFactor) + sizeof(SGLTFMaterialParameters::SPBRMetallicRoughness::metallicFactor) + sizeof(SGLTFMaterialParameters::SPBRMetallicRoughness::roughnessFactor) + sizeof(SGLTFMaterialParameters::emissiveFactor) + sizeof(SGLTFMaterialParameters::alphaMode) + sizeof(SGLTFMaterialParameters::alphaCutoff)), "Something went wrong");

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
