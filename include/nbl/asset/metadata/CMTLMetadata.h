// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_MTL_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_MTL_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl
{
namespace asset
{

class CMTLMetadata final : public IAssetMetadata
{
    public:
        class CIRenderpassIndependentPipeline : public IRenderpassIndependentPipelineMetadata
        {
            public:
                enum E_MAP_TYPE : uint32_t
                {
                    EMP_AMBIENT,
                    EMP_DIFFUSE,
                    EMP_SPECULAR,
                    EMP_EMISSIVE,
                    EMP_SHININESS,
                    EMP_OPACITY,
                    EMP_BUMP,
                    EMP_NORMAL,
                    EMP_DISPLACEMENT,
                    EMP_ROUGHNESS,
                    EMP_METALLIC,
                    EMP_SHEEN,
                    EMP_REFL_POSX,
                    EMP_REFL_NEGX,
                    EMP_REFL_POSY,
                    EMP_REFL_NEGY,
                    EMP_REFL_POSZ,
                    EMP_REFL_NEGZ,

                    EMP_COUNT
                };
                #include "nbl/nblpack.h"
                //! This struct is compliant with GLSL's std140 and std430 layouts
                struct alignas(16) SMaterialParameters
                {
                    //Ka
                    core::vector3df_SIMD ambient = core::vector3df_SIMD(1.f);
                    //Kd
                    core::vector3df_SIMD diffuse = core::vector3df_SIMD(1.f);
                    //Ks
                    core::vector3df_SIMD specular = core::vector3df_SIMD(1.f);
                    //Ke
                    core::vector3df_SIMD emissive = core::vector3df_SIMD(1.f);
                    //Tf
                    core::vector3df_SIMD transmissionFilter = core::vector3df_SIMD(1.f);
                    //Ns, specular exponent in phong model
                    float shininess = 32.f;
                    //d
                    float opacity = 1.f;
                    //-bm
                    float bumpFactor = 1.f;

                    //PBR
                    //Ni, index of refraction
                    float IoR = 1.6f;
                    //Pr
                    float roughness = 0.f;
                    //Pm
                    float metallic = 0.f;
                    //Ps
                    float sheen = 0.f;
                    //Pc
                    float clearcoatThickness = 0.f;
                    //Pcr
                    float clearcoatRoughness = 0.f;
                    //aniso
                    float anisotropy = 0.f;
                    //anisor
                    float anisoRotation = 0.f;
                    //illum - bits [0;3]
                    //map presence: bits [4;16], order in accordance with E_MAP_TYPE
                    uint32_t extra = 0u;
                } PACK_STRUCT;
                #include "nbl/nblunpack.h"
                //VS Intellisense shows error here because it think vectorSIMDf is 32 bytes, but it just Intellisense - it'll build anyway
                static_assert(sizeof(SMaterialParameters) == 128ull, "Something went wrong");


                using IRenderpassIndependentPipelineMetadata::IRenderpassIndependentPipelineMetadata;

                inline CIRenderpassIndependentPipeline& operator=(CIRenderpassIndependentPipeline&& other)
                {
                    IRenderpassIndependentPipelineMetadata::operator=(std::move(other));
                    std::swap(m_hash,other.m_hash);
                    return *this;
                }

                inline bool usesShaderWithUVs() const { return m_hash & 0x1u; }


                core::smart_refctd_ptr<ICPUDescriptorSet> m_descriptorSet3;
                SMaterialParameters m_materialParams;
                std::string m_name;
                //for permutations of pipeline representing same material but with different factors impossible to know from MTL file (like whether submesh using the material contains UVs)
                uint32_t m_hash;
        };

        CMTLMetadata(uint32_t pplnCount) : IAssetMetadata(), m_metaStorage(createContainer<CIRenderpassIndependentPipeline>(pplnCount))
        {
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CGraphicsPipelineLoaderMTL";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        meta_container_smart_ptr_t<CIRenderpassIndependentPipeline> m_metaStorage;

        friend class CPLYMeshFileLoader;
        template<typename... Args>
        inline void addMeta(
            uint32_t offset, const ICPURenderpassIndependentPipeline* ppln,
            core::smart_refctd_ptr<ICPUDescriptorSet>&& _descriptorSet3,
            const CIRenderpassIndependentPipeline::SMaterialParameters& _materialParams,
            std::string&& _name, uint32_t _hash,
            Args&&... args)
        {
            auto& meta = m_metaStorage->operator[](offset);
            meta = CIRenderpassIndependentPipeline(std::forward(args)...);
            meta.m_descriptorSet3 = std::move(_descriptorSet3);
            meta.m_materialParams = _materialParams;
            meta.m_name = std::move(_name);
            meta.m_hash = _hash;

            IAssetMetadata::insertAssetSpecificMetadata(ppln, &meta);
        }
};

}
}

#endif
