#ifndef __IRR_C_MTL_PIPELINE_METADATA_H_INCLUDED__
#define __IRR_C_MTL_PIPELINE_METADATA_H_INCLUDED__

#include "irr/asset/IPipelineMetadata.h"

namespace irr {
namespace asset
{

class CMTLPipelineMetadata final : public IPipelineMetadata
{
public:
#include "irr/irrpack.h"
    //! This struct is compliant with GLSL's std140 and std430 layouts
    struct alignas(16) SMTLMaterialParameters
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
#include "irr/irrunpack.h"
    //VS Intellisense shows error here because it think vectorSIMDf is 32 bytes, but it just Intellisense - it'll build anyway
    static_assert(sizeof(SMTLMaterialParameters) == 128ull, "Something went wrong");

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

    CMTLPipelineMetadata(const SMTLMaterialParameters& _params, std::string&& _name, core::smart_refctd_ptr<ICPUDescriptorSet>&& _ds3, uint32_t _hash, core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs) :
        m_materialParams(_params), m_name(std::move(_name)), m_descriptorSet3(std::move(_ds3)), m_hash(_hash), m_shaderInputs(std::move(_inputs)) {}

    const SMTLMaterialParameters& getMaterialParams() const { return m_materialParams; }
    const std::string getMaterialName() const { return m_name; }

    core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override { return { m_shaderInputs->begin(), m_shaderInputs->end() }; }
    const char* getLoaderName() const override { return "CGraphicsPipelineLoaderMTL"; } //?? i dont really understand the docs specifying what this function should return

    uint32_t getHashVal() const { return m_hash; }
    ICPUDescriptorSet* getDescriptorSet() const { return m_descriptorSet3.get(); }

private:
    SMTLMaterialParameters m_materialParams;
    std::string m_name;
    core::smart_refctd_ptr<ICPUDescriptorSet> m_descriptorSet3;
    //for permutations of pipeline representing same material but with different factors impossible to know from MTL file (like whether submesh using the material contains UVs)
    uint32_t m_hash;
    core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
};

}}

#endif