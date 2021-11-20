#ifndef __NBL_C_GLTF_PIPELINE_METADATA_H_INCLUDED__
#define _NBL_C_GLTF_PIPELINE_METADATA_H_INCLUDED_

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/metadata/IRenderpassIndependentPipelineMetadata.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/asset/ICPUPipelineLayout.h"

namespace nbl
{
namespace asset
{

class CGLTFPipelineMetadata final : public IAssetMetadata, public IRenderpassIndependentPipelineMetadata
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

        #include "nbl/nblpack.h"
        //! This struct is compliant with GLSL's std140 and std430 layouts
        struct SGLTFMaterialParameters
        {
            struct SPBRMetallicRoughness
            {
                core::vector4df_SIMD baseColorFactor = core::vector4df_SIMD(1.f, 1.f, 1.f, 1.f); // why is base color vec4, does it need alpha?
                float metallicFactor = 1.f;
                float roughnessFactor = 1.f;
            };

            SPBRMetallicRoughness metallicRoughness;
            core::vector3df_SIMD emissiveFactor = core::vector3df_SIMD(0, 0, 0);
            E_ALPHA_MODE alphaMode = EAM_OPAQUE;
            float alphaCutoff = 0.5f; 

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

        } PACK_STRUCT;
        #include "nbl/nblunpack.h"

        static_assert(sizeof(SGLTFMaterialParameters) <= asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
            
        CGLTFPipelineMetadata(core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs)
            : IRenderpassIndependentPipelineMetadata(core::SRange<const IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>(_inputs->begin(),_inputs->end())) {}

        static inline constexpr const char* loaderName = "CGLTFLoader";
        const char* getLoaderName() const override { return loaderName; }
};

}
}

#endif // _NBL_C_GLTF_PIPELINE_METADATA_H_INCLUDED_
