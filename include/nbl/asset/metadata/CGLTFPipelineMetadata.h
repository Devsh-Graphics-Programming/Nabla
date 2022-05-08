#ifndef _NBL_C_GLTF_PIPELINE_METADATA_H_INCLUDED_
#define _NBL_C_GLTF_PIPELINE_METADATA_H_INCLUDED_

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/metadata/IRenderpassIndependentPipelineMetadata.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/asset/ICPUPipelineLayout.h"

namespace nbl::asset
{

class NBL_API CGLTFPipelineMetadata final : public IRenderpassIndependentPipelineMetadata
{
    public:
        CGLTFPipelineMetadata() {}
        virtual ~CGLTFPipelineMetadata() {}

        enum E_ALPHA_MODE : uint32_t
        {
            EAM_OPAQUE = core::createBitmask({0}),
            EAM_MASK = core::createBitmask({1}),
            EAM_BLEND = core::createBitmask({2})
        };

        //! This struct is compliant with GLSL's std140 and std430 layouts
        struct alignas(16) SGLTFMaterialParameters
        {
            struct SPBRMetallicRoughness
            {
                core::vector4df_SIMD baseColorFactor = core::vector4df_SIMD(1.f, 1.f, 1.f, 1.f); // TODO: why is base color vec4, does it need alpha?
                float metallicFactor = 1.f;
                float roughnessFactor = 1.f;
            };

            SPBRMetallicRoughness metallicRoughness;
            core::vector3df_SIMD emissiveFactor = core::vector3df_SIMD(0, 0, 0);
            E_ALPHA_MODE alphaMode = EAM_OPAQUE; // TODO: This can be removed!
            float alphaCutoff = 1.f; // should be 0.f for opaque, 0.5f for masked, 1.f/255.f for blend

            enum E_GLTF_TEXTURES : uint32_t
            {
                EGT_BASE_COLOR_TEXTURE = core::createBitmask<uint8_t>({ 0 }),
                EGT_METALLIC_ROUGHNESS_TEXTURE = core::createBitmask<uint8_t>({ 1 }),
                EGT_NORMAL_TEXTURE = core::createBitmask<uint8_t>({ 2 }),
                EGT_OCCLUSION_TEXTURE = core::createBitmask<uint8_t>({ 3 }),
                EGT_EMISSIVE_TEXTURE = core::createBitmask<uint8_t>({ 4 }),
                EGT_COUNT = 5,
            };

            uint32_t availableTextures = 0;

        };

        static_assert(sizeof(SGLTFMaterialParameters) <= asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
            
        CGLTFPipelineMetadata(core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
            : IRenderpassIndependentPipelineMetadata(core::SRange<const IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>(_inputs->begin(),_inputs->end())) {}
};

}

#endif // _NBL_C_GLTF_PIPELINE_METADATA_H_INCLUDED_
