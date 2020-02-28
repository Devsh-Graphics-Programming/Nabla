#include "irr/asset/CGraphicsPipelineLoaderMTL.h"

#include "irr/asset/IBuiltinIncludeLoader.h"
namespace irr {
namespace asset
{    

class CGLSL_MTLBuiltinIncludeLoader : public IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/graphicspipeline/loaders/mtl/"; }

private:
    static std::string getMTLloaderCommonDefinitions(const std::string&)
    {
        return
R"(#ifndef _IRR_MTL_LOADER_COMMON_INCLUDED_
#define _IRR_MTL_LOADER_COMMON_INCLUDED_

struct irr_glsl_MTLMaterialParameters
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    vec4 Tf;//w component doesnt matter
    float Ns;
    float d;
    float bm;
    float Ni;
    float roughness;
    float metallic;
    float sheen;
    float clearcoatThickness;
    float clearcoatRoughness;
    float anisotropy;
    float anisoRotation;
    //extra info
    uint extra;
};

#endif
)";
    }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"common\\.glsl"}, &getMTLloaderCommonDefinitions },
        };
    }
};

}}

namespace
{
/*
    constexpr const char* FRAG_SHADER_NO_UV_PBR =
R"(#version 430 core

layout (location = 0) in vec3 LocalPos;
layout (location = 1) in vec3 ViewPos;
layout (location = 2) in vec3 Normal;
layout (location = 0) out vec4 OutColor;

layout (push_constant) uniform Block {
    vec3 ambient;
    vec3 albedo;//MTL's diffuse
    vec3 specular;
    vec3 emissive;
    vec4 Tf;//w component doesnt matter
    float shininess;
    float opacity;
    float bumpFactor;
    //PBR
    float ior;
    float roughness;
    float metallic;
    float sheen;
    float clearcoatThickness;
    float clearcoatRoughness;
    float anisotropy;
    float anisoRotation;
    //extra info
    uint extra;
} PC;

#define PI 3.14159265359
#define FLT_MIN 1.175494351e-38

#include <irr/builtin/glsl/bsdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/ndf/ggx_trowbridge_reitz.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/geom/ggx_smith.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>

void main()
{
    vec3 N = normalize(Normal);
    //some approximation for computing tangents without UV
    vec3 c1 = cross(N, vec3(0.0, 0.0, 1.0));
    vec3 c2 = cross(N, vec3(0.0, 1.0, 0.0));
    vec3 T = (dot(c1,c1) > dot(c2,c2)) ? c1 : c2;
    T = normalize(T);
    vec3 B = normalize(cross(N,T));
    vec3 V = -ViewPos;

    vec3 NdotV = dot(N,V);
#define NdotL NdotV
#define NdotH NdotV

    vec3 color = PC.params.emissive*0.01;
    if (NdotL > FLT_MIN)
    {
        float lightDistance2 = dot(V,V);
        float Vrcplen = inversesqrt(lightDistance2);
        NdotV *= Vrcplen;
        V *= Vrcplen;

        vec3 TdotV = dot(T,V);
        vec3 BdotV = dot(B,V);
#define TdotL TdotV
#define BdotL BdotV
#define TdotH TdotV
#define BdotH BdotV

        float at = sqrt(PC.params.roughness);
        float ab = at*(1.0 - PC.params.anisotropy);

        float fr = irr_glsl_fresnel_dielectric(PC.params.ior, NdotV);
        float one_minus_fr = 1.0-fr;
        float diffuseFactor = 1.0 - one_minus_fr*one_minus_fr;
        float diffuse = 0.0;
        if (PC.params.metallic < 1.0)
        {
            if (PC.params.roughness==0.0)
                diffuse = 1.0/PI;
            else
                diffuse = oren_nayar(PC.params.roughness, N, V, V, NdotL, NdotV);
        }
        float specular = 0.0;
        if (NdotV > FLT_MIN)
        {
            float ndf = GGXBurleyAnisotropic(PC.params.anisotropy, PC.params.roughness, TdotH, BdotH, NdotH);
            float geom = GGXSmithHeightCorrelated_aniso_wo_numerator(at, ab, TdotL, TdotV, BdotL, BdotV, NdotL, NdotV);
            specular = ndf*geom*fr;
        }

        color += (diffuseFactor*diffuse*PC.params.albedo + specular) * NdotL / lightDistance2;
    }
    OutColor = vec4(color*PC.params.transmissionFilter, 1.0);
}
)";
*/
    constexpr const char* VERT_SHADER_UV = 
R"(#version 430 core

#ifndef _IRR_VERT_INPUTS_DEFINED_
#define _IRR_VERT_INPUTS_DEFINED_
layout (location = 0) in vec3 vPos;
#ifndef _NO_UV
layout (location = 2) in vec2 vUV;
#endif
layout (location = 3) in vec3 vNormal;
#endif //_IRR_VERT_INPUTS_DEFINED_

#ifndef _IRR_VERT_OUTPUTS_DEFINED_
#define _IRR_VERT_OUTPUTS_DEFINED_
layout (location = 0) out vec3 LocalPos;
layout (location = 1) out vec3 ViewPos;
layout (location = 2) out vec3 Normal;
#ifndef _NO_UV
layout (location = 3) out vec2 UV;
#endif
#endif //_IRR_VERT_OUTPUTS_DEFINED_

#include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>

#ifndef _IRR_VERT_SET1_BINDINGS_DEFINED_
#define _IRR_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    irr_glsl_SBasicViewParameters params;
} CamData;
#endif //_IRR_VERT_SET1_BINDINGS_DEFINED_

#ifndef _IRR_VERT_MAIN_DEFINED_
#define _IRR_VERT_MAIN_DEFINED_
void main()
{
    LocalPos = vPos;
    gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(CamData.params.MVP), vPos);
    ViewPos = irr_glsl_pseudoMul3x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x3(CamData.params.MV), vPos);
    mat3 normalMat = irr_glsl_SBasicViewParameters_GetNormalMat(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x3(CamData.params.NormalMatAndEyePos));
    Normal = normalMat*normalize(vNormal);
#ifndef _NO_UV
    UV = vUV;
#endif
}
#endif //_IRR_VERT_MAIN_DEFINED_
)";
    constexpr const char* FRAG_SHADER_UV =
R"(#version 430 core

#ifndef _IRR_FRAG_INPUTS_DEFINED_
#define _IRR_FRAG_INPUTS_DEFINED_
layout (location = 0) in vec3 LocalPos;
layout (location = 1) in vec3 ViewPos;
layout (location = 2) in vec3 Normal;
#ifndef _NO_UV
layout (location = 3) in vec2 UV;
#endif
#endif //_IRR_FRAG_INPUTS_DEFINED_

#ifndef _IRR_FRAG_OUTPUTS_DEFINED_
#define _IRR_FRAG_OUTPUTS_DEFINED_
layout (location = 0) out vec4 OutColor;
#endif //_IRR_FRAG_OUTPUTS_DEFINED_

#define ILLUM_MODEL_MASK 0x0fu
#define map_Ka_MASK uint(1u<<4u)
#define map_Kd_MASK uint(1u<<5u)
#define map_Ks_MASK uint(1u<<6u)
#define map_Ns_MASK uint(1u<<8u)
#define map_d_MASK uint(1u<<9u)
#define map_bump_MASK uint(1u<<10u)
#define map_normal_MASK uint(1u<<11u)

#include <irr/builtin/glsl/bsdf/common.glsl>

#ifndef _IRR_FRAG_PUSH_CONSTANTS_DEFINED_
#define _IRR_FRAG_PUSH_CONSTANTS_DEFINED_

#include <irr/builtin/glsl/graphicspipeline/loaders/mtl/common.glsl>

layout (push_constant) uniform Block {
    irr_glsl_MTLMaterialParameters params;
} PC;
#endif //_IRR_FRAG_PUSH_CONSTANTS_DEFINED_

#if !defined(_IRR_FRAG_SET3_BINDINGS_DEFINED_) && !defined(_NO_UV)
#define _IRR_FRAG_SET3_BINDINGS_DEFINED_
layout (set = 3, binding = 0) uniform sampler2D map_Ka;
layout (set = 3, binding = 1) uniform sampler2D map_Kd;
layout (set = 3, binding = 2) uniform sampler2D map_Ks;
layout (set = 3, binding = 4) uniform sampler2D map_Ns;
layout (set = 3, binding = 5) uniform sampler2D map_d;
layout (set = 3, binding = 6) uniform sampler2D map_bump;
#endif //_IRR_FRAG_SET3_BINDINGS_DEFINED_

#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>
#include <irr/builtin/glsl/bsdf/brdf/diffuse/fresnel_correction.glsl>
#include <irr/builtin/glsl/bsdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/blinn_phong.glsl>

#ifndef _IRR_BSDF_COS_EVAL_DEFINED_
#define _IRR_BSDF_COS_EVAL_DEFINED_

// Spectrum can be exchanged to a float for monochrome
#define Spectrum vec3

//! This is the function that evaluates the BSDF for specific view and observer direction
// params can be either BSDFIsotropicParams or BSDFAnisotropicParams 
Spectrum irr_bsdf_cos_eval(in irr_glsl_BSDFIsotropicParams params)
{
    vec3 Kd;
#ifndef _NO_UV
    if ((PC.params.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Kd = texture(map_Kd, UV).rgb;
    else
#endif
        Kd = PC.params.Kd;

    vec3 color = vec3(0.0);
    vec3 Ks;
    float Ns;

#ifndef _NO_UV
    if ((PC.params.extra&(map_Ks_MASK)) == (map_Ks_MASK))
        Ks = texture(map_Ks, UV).rgb;
    else
#endif
        Ks = PC.params.Ks;
#ifndef _NO_UV
    if ((PC.params.extra&(map_Ns_MASK)) == (map_Ns_MASK))
        Ns = texture(map_Ns, UV).x;
    else
#endif
        Ns = PC.params.Ns;

    vec3 Ni = vec3(PC.params.Ni);

    vec3 diff = irr_glsl_lambertian_cos_eval(params) * Kd * (1.0-irr_glsl_fresnel_dielectric(Ni,params.NdotL)) * (1.0-irr_glsl_fresnel_dielectric(Ni,params.NdotV));
    diff *= irr_glsl_diffuseFresnelCorrectionFactor(Ni, Ni*Ni);
    switch (PC.params.extra&ILLUM_MODEL_MASK)
    {
    case 0:
        color = vec3(0.0);
        break;
    case 1:
        color = diff;
        break;
    case 2:
    case 3://2 + IBL
    case 5://basically same as 3
    case 8://basically same as 5
    {
        vec3 spec = Ks*irr_glsl_blinn_phong_fresnel_dielectric_cos_eval(params, Ns, Ni);
        color = (diff + spec);
    }
        break;
    case 4:
    case 6:
    case 7:
    case 9://basically same as 4
    {
        vec3 spec = Ks*irr_glsl_blinn_phong_fresnel_dielectric_cos_eval(params, Ns, Ni);
        color = spec;
    }
        break;
    default:
        break;
    }

    return color;  
}

#endif //_IRR_BSDF_COS_EVAL_DEFINED_

#include <irr/builtin/glsl/bump_mapping/height_mapping.glsl>

#ifndef _IRR_COMPUTE_LIGHTING_DEFINED_
#define _IRR_COMPUTE_LIGHTING_DEFINED_

vec3 irr_computeLighting(out irr_glsl_ViewSurfaceInteraction out_interaction)
{
    irr_glsl_ViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(vec3(0.0), ViewPos, Normal);

#ifndef _NO_UV
    if ((PC.params.extra&map_bump_MASK) == map_bump_MASK)
    {
        interaction.N = normalize(interaction.N);

        float h = texture(map_bump, UV).x;

        vec2 dHdScreen = vec2(dFdx(h), dFdy(h));
        interaction.N = irr_glsl_perturbNormal_heightMap(interaction.N, interaction.V.dPosdScreen, dHdScreen);
    }
#endif
    irr_glsl_BSDFIsotropicParams params = irr_glsl_calcBSDFIsotropicParams(interaction, -ViewPos);

    vec3 Ka;
    switch ((PC.params.extra&ILLUM_MODEL_MASK))
    {
    case 0:
    {
#ifndef _NO_UV
    if ((PC.params.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Ka = texture(map_Kd, UV).rgb;
    else
#endif
        Ka = PC.params.Kd;
    }
    break;
    default:
#define Ia 0.1
    {
#ifndef _NO_UV
    if ((PC.params.extra&(map_Ka_MASK)) == (map_Ka_MASK))
        Ka = texture(map_Ka, UV).rgb;
    else
#endif
        Ka = PC.params.Ka;
    Ka *= Ia;
    }
#undef Ia
    break;
    }

    out_interaction = params.interaction;
#define Intensity 1000.0
    return Intensity*params.invlenL2*irr_bsdf_cos_eval(params) + Ka;
#undef Intensity
}
#endif //_IRR_COMPUTE_LIGHTING_DEFINED_


#ifndef _IRR_FRAG_MAIN_DEFINED_
#define _IRR_FRAG_MAIN_DEFINED_

void main()
{
    irr_glsl_ViewSurfaceInteraction interaction;
    vec3 color = irr_computeLighting(interaction);

    float d = PC.params.d;

    //another illum model switch, required for illum=4,6,7,9 to compute alpha from fresnel (taken from opacity map or constant otherwise)
    switch (PC.params.extra&ILLUM_MODEL_MASK)
    {
    case 4:
    case 6:
    case 7:
    case 9:
    {
        float VdotN = dot(interaction.N, interaction.V.dir);
        d = irr_glsl_fresnel_dielectric(vec3(PC.params.Ni), VdotN).x;
    }
        break;
    default:
#ifndef _NO_UV
        if ((PC.params.extra&(map_d_MASK)) == (map_d_MASK))
        {
            d = texture(map_d, UV).r;
            color *= d;
        }
#endif
        break;
    }

    OutColor = vec4(color, d);
}
#endif //_IRR_FRAG_MAIN_DEFINED_
)";
}


using namespace irr;
using namespace asset;

static void insertShaderIntoCache(core::smart_refctd_ptr<ICPUSpecializedShader>& asset, const char* path, IAssetManager* _assetMgr)
{
    asset::SAssetBundle bundle({ asset });
    _assetMgr->changeAssetKey(bundle, path);
    _assetMgr->insertAssetIntoCache(bundle);
}

_IRR_STATIC_INLINE_CONSTEXPR const char* VERT_SHADER_NO_UV_CACHE_KEY = "irr/builtin/shader/loaders/mtl/vertex_no_uv";
_IRR_STATIC_INLINE_CONSTEXPR const char* VERT_SHADER_UV_CACHE_KEY = "irr/builtin/shader/loaders/mtl/vertex_uv";
_IRR_STATIC_INLINE_CONSTEXPR const char* FRAG_SHADER_NO_UV_CACHE_KEY = "irr/builtin/shader/loaders/mtl/fragment_no_uv";
_IRR_STATIC_INLINE_CONSTEXPR const char* FRAG_SHADER_UV_CACHE_KEY = "irr/builtin/shader/loaders/mtl/fragment_uv";

CGraphicsPipelineLoaderMTL::CGraphicsPipelineLoaderMTL(IAssetManager* _am) : m_assetMgr{_am}
{
    m_assetMgr->getGLSLCompiler()->getIncludeHandler()->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<CGLSL_MTLBuiltinIncludeLoader>());

    //create vertex shaders and insert them into cache
    {
        std::string vs_nouv_source = VERT_SHADER_UV;
        vs_nouv_source.insert(vs_nouv_source.find("\n")+1ull, "#define _NO_UV\n"); 

        auto vs_nouv_unspec = core::make_smart_refctd_ptr<ICPUShader>(vs_nouv_source.c_str());
        auto vs_uv_unspec = core::make_smart_refctd_ptr<ICPUShader>(VERT_SHADER_UV);

        ICPUSpecializedShader::SInfo specinfo({}, nullptr, "main", ICPUSpecializedShader::ESS_VERTEX);
        auto vs_nouv = core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(vs_nouv_unspec), ICPUSpecializedShader::SInfo(specinfo));
        auto vs_uv = core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(vs_uv_unspec), std::move(specinfo));

        insertShaderIntoCache(vs_nouv, VERT_SHADER_NO_UV_CACHE_KEY, m_assetMgr);
        insertShaderIntoCache(vs_uv, VERT_SHADER_UV_CACHE_KEY, m_assetMgr);
    }
    //create fragment shaders and insert them into cache
    {
        std::string fs_nouv_source = FRAG_SHADER_UV;
        fs_nouv_source.insert(fs_nouv_source.find('\n') + 1ull,
            R"(
#ifndef _IRR_FRAG_SET3_BINDINGS_DEFINED_
#define _IRR_FRAG_SET3_BINDINGS_DEFINED_
#endif
)"
);
        fs_nouv_source.insert(fs_nouv_source.find("\n")+1ull, "#define _NO_UV\n");

        auto fs_nouv_unspec = core::make_smart_refctd_ptr<ICPUShader>(fs_nouv_source.c_str());
        auto fs_uv_unspec = core::make_smart_refctd_ptr<ICPUShader>(FRAG_SHADER_UV);

        ICPUSpecializedShader::SInfo specinfo({}, nullptr, "main", ICPUSpecializedShader::ESS_FRAGMENT);
        auto fs_nouv = core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(fs_nouv_unspec), ICPUSpecializedShader::SInfo(specinfo));
        auto fs_uv = core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(fs_uv_unspec), std::move(specinfo));

        insertShaderIntoCache(fs_nouv, FRAG_SHADER_NO_UV_CACHE_KEY, m_assetMgr);
        insertShaderIntoCache(fs_uv, FRAG_SHADER_UV_CACHE_KEY, m_assetMgr);
    }
}

bool CGraphicsPipelineLoaderMTL::isALoadableFileFormat(io::IReadFile* _file) const
{
    if (!_file)
        return false;

    const size_t prevPos = _file->getPos();

    _file->seek(0ull);

    std::string mtl;
    mtl.resize(_file->getSize());
    _file->read(mtl.data(), _file->getSize());
    _file->seek(prevPos);

    return mtl.find("newmtl") != std::string::npos;
}

template<typename AssetType, IAsset::E_TYPE assetType>
static core::smart_refctd_ptr<AssetType> getDefaultAsset(const char* _key, IAssetManager* _assetMgr)
{
    size_t storageSz = 1ull;
    asset::SAssetBundle bundle;
    const IAsset::E_TYPE types[]{ assetType, static_cast<IAsset::E_TYPE>(0u) };

    _assetMgr->findAssets(storageSz, &bundle, _key, types);
    if (bundle.isEmpty())
        return nullptr;
    auto assets = bundle.getContents();
    //assert(assets.first != assets.second);

    return core::smart_refctd_ptr_static_cast<AssetType>(assets.first[0]);
}

core::smart_refctd_ptr<ICPUPipelineLayout> CGraphicsPipelineLoaderMTL::makePipelineLayoutFromMtl(const SMtl& _mtl, bool _noDS3)
{
    //assumes all supported textures are always present
    //since vulkan doesnt support bindings with no/null descriptor, absent textures will be filled with dummy 2D texture (while creating desc set)
    auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUDescriptorSetLayout::SBinding>>(static_cast<size_t>(CMTLPipelineMetadata::EMP_REFL_POSX)+1ull);

    ICPUDescriptorSetLayout::SBinding bnd;
    bnd.count = 1u;
    bnd.stageFlags = ICPUSpecializedShader::ESS_FRAGMENT;
    bnd.type = EDT_COMBINED_IMAGE_SAMPLER;
    bnd.binding = 0u;
    std::fill(bindings->begin(), bindings->end(), bnd);

    core::smart_refctd_ptr<ICPUSampler> samplers[2];
    samplers[0] = getDefaultAsset<ICPUSampler,IAsset::ET_SAMPLER>("irr/builtin/samplers/default", m_assetMgr);
    samplers[1] = getDefaultAsset<ICPUSampler, IAsset::ET_SAMPLER>("irr/builtin/samplers/default_clamp_to_border", m_assetMgr);
    for (uint32_t i = 0u; i <= CMTLPipelineMetadata::EMP_REFL_POSX; ++i)
    {
        (*bindings)[i].binding = i;

        const uint32_t clamp = (_mtl.clamp >> i) & 1u;
        (*bindings)[i].samplers = samplers + clamp;
    }

    auto ds1layout = getDefaultAsset<ICPUDescriptorSetLayout, IAsset::ET_DESCRIPTOR_SET_LAYOUT>("irr/builtin/descriptor_set_layout/basic_view_parameters", m_assetMgr);
    core::smart_refctd_ptr<ICPUDescriptorSetLayout> ds3Layout = _noDS3 ? nullptr : core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings->begin(), bindings->end());
    SPushConstantRange pcRng;
    pcRng.stageFlags = ICPUSpecializedShader::ESS_FRAGMENT;
    pcRng.offset = 0u;
    pcRng.size = sizeof(SMtl::params);
    //if intellisense shows error here, it's most likely intellisense's fault and it'll build fine anyway
    static_assert(sizeof(SMtl::params)<=ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "It must fit in push constants!");
    //ds with textures for material goes to set=3
    auto layout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(&pcRng, &pcRng+1, nullptr, std::move(ds1layout), nullptr, std::move(ds3Layout));

    return layout;
}

SAssetBundle CGraphicsPipelineLoaderMTL::loadAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    constexpr uint32_t POSITION = 0u;
    constexpr uint32_t UV = 2u;
    constexpr uint32_t NORMAL = 3u;
    constexpr uint32_t BND_NUM = 0u;

    SContext ctx(
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        },
        _hierarchyLevel,
        _override
    );
    const io::path fullName = _file->getFileName();
	const std::string relPath = (io::IFileSystem::getFileDir(fullName)+"/").c_str();

    auto materials = readMaterials(_file);

    constexpr uint32_t PIPELINE_PERMUTATION_COUNT = 2u;

    core::vector<core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>> pipelines(materials.size()*PIPELINE_PERMUTATION_COUNT);
    for (size_t i = 0ull; i < materials.size(); ++i)
    {
        SVertexInputParams vtxParams;
        SBlendParams blendParams;
        SPrimitiveAssemblyParams primParams;
        SRasterizationParams rasterParams;

        const uint32_t illum = materials[i].params.extra&0xfu;
        if (illum==4u || illum==6u || illum==7u || illum==9u)
        {
            blendParams.blendParams[0].blendEnable = true;
            blendParams.blendParams[0].srcColorFactor = EBF_ONE;
            blendParams.blendParams[0].srcAlphaFactor = EBF_ONE;
            blendParams.blendParams[0].dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
            blendParams.blendParams[0].dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
        }
        else if (materials[i].maps[CMTLPipelineMetadata::EMP_OPACITY].size() || materials[i].params.opacity!=1.f)
        {
            blendParams.blendParams[0].blendEnable = true;
            blendParams.blendParams[0].srcColorFactor = EBF_SRC_ALPHA;
            blendParams.blendParams[0].srcAlphaFactor = EBF_SRC_ALPHA;
            blendParams.blendParams[0].dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
            blendParams.blendParams[0].dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
        }

        const uint32_t j = i*PIPELINE_PERMUTATION_COUNT;

        vtxParams.enabledAttribFlags = (1u << POSITION) | (1u << NORMAL);
        vtxParams.enabledBindingFlags = 1u << BND_NUM;
        vtxParams.bindings[BND_NUM].stride = 24u;
        vtxParams.bindings[BND_NUM].inputRate = EVIR_PER_VERTEX;
        //position
        vtxParams.attributes[POSITION].binding = BND_NUM;
        vtxParams.attributes[POSITION].format = EF_R32G32B32_SFLOAT;
        vtxParams.attributes[POSITION].relativeOffset = 0u;
        //normal
        vtxParams.attributes[NORMAL].binding = BND_NUM;
        vtxParams.attributes[NORMAL].format = EF_A2B10G10R10_SNORM_PACK32;
        vtxParams.attributes[NORMAL].relativeOffset = 20u;

        auto layout = makePipelineLayoutFromMtl(materials[i], true);
        auto shaders = getShaders(false);

        constexpr size_t DS1_METADATA_ENTRY_CNT = 3ull;
        core::smart_refctd_dynamic_array<IPipelineMetadata::ShaderInputSemantic> shaderInputsMetadata = core::make_refctd_dynamic_array<decltype(shaderInputsMetadata)>(DS1_METADATA_ENTRY_CNT);
        {
            ICPUDescriptorSetLayout* ds1layout = layout->getDescriptorSetLayout(1u);

            constexpr IPipelineMetadata::E_COMMON_SHADER_INPUT types[DS1_METADATA_ENTRY_CNT]{IPipelineMetadata::ECSI_WORLD_VIEW_PROJ, IPipelineMetadata::ECSI_WORLD_VIEW, IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE};
            constexpr uint32_t sizes[DS1_METADATA_ENTRY_CNT]{sizeof(SBasicViewParameters::MVP), sizeof(SBasicViewParameters::MV), sizeof(SBasicViewParameters::NormalMat)};
            constexpr uint32_t relOffsets[DS1_METADATA_ENTRY_CNT]{offsetof(SBasicViewParameters,MVP), offsetof(SBasicViewParameters,MV), offsetof(SBasicViewParameters,NormalMat)};
            for (uint32_t i = 0u; i < DS1_METADATA_ENTRY_CNT; ++i)
            {
                auto& semantic = (shaderInputsMetadata->end()-i-1u)[0];
                semantic.type = types[i];
                semantic.descriptorSection.type = IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER;
                semantic.descriptorSection.uniformBufferObject.binding = ds1layout->getBindings().begin()[0].binding;
                semantic.descriptorSection.uniformBufferObject.set = 1u;
                semantic.descriptorSection.uniformBufferObject.relByteoffset = relOffsets[i];
                semantic.descriptorSection.uniformBufferObject.bytesize = sizes[i];
                semantic.descriptorSection.shaderAccessFlags = ICPUSpecializedShader::ESS_VERTEX;
            }
        }

        pipelines[j] = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(layout), nullptr, nullptr, vtxParams, blendParams, primParams, rasterParams);
        pipelines[j]->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, shaders.first.get());
        pipelines[j]->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, shaders.second.get());
        m_assetMgr->setAssetMetadata(pipelines[j].get(), core::make_smart_refctd_ptr<CMTLPipelineMetadata>(materials[i].params, std::string(materials[i].name), nullptr, 0u, core::smart_refctd_ptr(shaderInputsMetadata)));

        //uv
        vtxParams.enabledAttribFlags |= (1u << UV);
        vtxParams.attributes[UV].binding = BND_NUM;
        vtxParams.attributes[UV].format = EF_R32G32_SFLOAT;
        vtxParams.attributes[UV].relativeOffset = 12u;

        layout = makePipelineLayoutFromMtl(materials[i], false);
        shaders = getShaders(true);

        core::smart_refctd_ptr<ICPUDescriptorSet> ds3;
        {
            const std::string dsCacheKey = std::string(fullName.c_str())+"?"+materials[i].name+"?_ds";
            const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_DESCRIPTOR_SET, (asset::IAsset::E_TYPE)0u };
            auto ds_bundle = _override->findCachedAsset(dsCacheKey, types, ctx.inner, _hierarchyLevel + ICPUMesh::DESC_SET_HIERARCHYLEVELS_BELOW);
            if (!ds_bundle.isEmpty())
                ds3 = core::smart_refctd_ptr_static_cast<ICPUDescriptorSet>(ds_bundle.getContents().first[0]);
            else 
            {
                auto views = loadImages(relPath.c_str(), materials[i], ctx);
                ds3 = makeDescSet(std::move(views), layout->getDescriptorSetLayout(3u));
                if (ds3)
                {
                    SAssetBundle bundle{ds3};
                    _override->insertAssetIntoCache(bundle, dsCacheKey, ctx.inner, _hierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
                }
            }
        }

        pipelines[j+1u] = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(layout), nullptr, nullptr, vtxParams, blendParams, primParams, rasterParams);
        pipelines[j+1u]->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, shaders.first.get());
        pipelines[j+1u]->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, shaders.second.get());
        m_assetMgr->setAssetMetadata(pipelines[j+1u].get(), core::make_smart_refctd_ptr<CMTLPipelineMetadata>(materials[i].params, std::move(materials[i].name), std::move(ds3), 1u, std::move(shaderInputsMetadata)));
    }
    materials.clear();

    return asset::SAssetBundle(std::move(pipelines));
}

namespace
{
    //! skip space characters and stop on first non-space
    const char* goFirstWord(const char* buf, const char* const _bufEnd, bool acrossNewlines = true)
    {
        // skip space characters
        if (acrossNewlines)
            while ((buf != _bufEnd) && core::isspace(*buf))
                ++buf;
        else
            while ((buf != _bufEnd) && core::isspace(*buf) && (*buf != '\n'))
                ++buf;

        return buf;
    }


    //! skip current word and stop at beginning of next one
    const char* goNextWord(const char* buf, const char* const _bufEnd, bool acrossNewlines = true)
    {
        // skip current word
        while ((buf != _bufEnd) && !core::isspace(*buf))
            ++buf;

        return goFirstWord(buf, _bufEnd, acrossNewlines);
    }


    //! Read until line break is reached and stop at the next non-space character
    const char* goNextLine(const char* buf, const char* const _bufEnd)
    {
        // look for newline characters
        while (buf != _bufEnd)
        {
            // found it, so leave
            if (*buf == '\n' || *buf == '\r')
                break;
            ++buf;
        }
        return goFirstWord(buf, _bufEnd);
    }


    uint32_t copyWord(char* outBuf, const char* const inBuf, uint32_t outBufLength, const char* const _bufEnd)
    {
        if (!outBufLength)
            return 0;
        if (!inBuf)
        {
            *outBuf = 0;
            return 0;
        }

        uint32_t i = 0;
        while (inBuf[i])
        {
            if (core::isspace(inBuf[i]) || &(inBuf[i]) == _bufEnd)
                break;
            ++i;
        }

        uint32_t length = core::min(i, outBufLength - 1u);
        for (uint32_t j = 0u; j < length; ++j)
            outBuf[j] = inBuf[j];

        outBuf[length] = 0;
        return length;
    }

    const char* goAndCopyNextWord(char* outBuf, const char* inBuf, uint32_t outBufLength, const char* _bufEnd)
    {
        inBuf = goNextWord(inBuf, _bufEnd, false);
        copyWord(outBuf, inBuf, outBufLength, _bufEnd);
        return inBuf;
    }
}

const char* CGraphicsPipelineLoaderMTL::readTexture(const char* _bufPtr, const char* const _bufEnd, SMtl* _currMaterial, const char* _mapType) const
{
    static const core::unordered_map<std::string, CMTLPipelineMetadata::E_MAP_TYPE> str2type =
    {
        {"Ka", CMTLPipelineMetadata::EMP_AMBIENT},
        {"Kd", CMTLPipelineMetadata::EMP_DIFFUSE},
        {"Ke", CMTLPipelineMetadata::EMP_EMISSIVE},
        {"Ks", CMTLPipelineMetadata::EMP_SPECULAR},
        {"Ns", CMTLPipelineMetadata::EMP_SHININESS},
        {"d", CMTLPipelineMetadata::EMP_OPACITY},
        {"bump", CMTLPipelineMetadata::EMP_BUMP},
        {"disp", CMTLPipelineMetadata::EMP_DISPLACEMENT},
        {"refl", CMTLPipelineMetadata::EMP_REFL_POSX},
        {"norm", CMTLPipelineMetadata::EMP_NORMAL},
        {"Pr", CMTLPipelineMetadata::EMP_ROUGHNESS},
        {"Pm", CMTLPipelineMetadata::EMP_METALLIC},
        {"Ps", CMTLPipelineMetadata::EMP_SHEEN}
    };
    static const core::unordered_map<std::string, CMTLPipelineMetadata::E_MAP_TYPE> refl_str2type =
    {
        {"top", CMTLPipelineMetadata::EMP_REFL_POSY},
        {"bottom", CMTLPipelineMetadata::EMP_REFL_NEGY},
        {"front", CMTLPipelineMetadata::EMP_REFL_NEGZ},
        {"back", CMTLPipelineMetadata::EMP_REFL_POSZ},
        {"left", CMTLPipelineMetadata::EMP_REFL_NEGX},
        {"right", CMTLPipelineMetadata::EMP_REFL_POSX}
    };

    constexpr static size_t WORD_BUFFER_LENGTH = 512ull;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

    std::string mapTypeStr = _mapType;
    if (mapTypeStr.compare(0ull, 4ull, "map_")==0)
        mapTypeStr.erase(0ull, 4ull);

    CMTLPipelineMetadata::E_MAP_TYPE mapType = CMTLPipelineMetadata::EMP_COUNT;
    auto found = str2type.find(mapTypeStr);
    if (found != str2type.end())
        mapType = found->second;

    constexpr uint32_t ILLUM_MODEL_BITS = 4u;
    _currMaterial->params.extra |= (1u << (ILLUM_MODEL_BITS + mapType));

    _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
    while (tmpbuf[0]=='-')
    {
        if (mapType==CMTLPipelineMetadata::EMP_REFL_POSX && strncmp(tmpbuf, "-type", 5)==0)
        {
            _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
            if (strlen(tmpbuf) >= 8ull) //shortest one is "cube_top"
            {
                found = refl_str2type.find(tmpbuf+5); //skip "cube_"
                if (found != refl_str2type.end())
                    mapType = found->second;
            }
        }
        else if (strncmp(_bufPtr,"-bm",3)==0)
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			sscanf(tmpbuf, "%f", &_currMaterial->params.bumpFactor);
		}
		else
		if (strncmp(_bufPtr,"-blendu",7)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-blendv",7)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-cc",3)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-clamp",6)==0)
        {
            _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
            if (mapType != CMTLPipelineMetadata::EMP_COUNT)
            {
                uint32_t clamp = (strcmp("off", tmpbuf) != 0);
                _currMaterial->clamp |= (clamp<<mapType);
            }
        }
		else
		if (strncmp(_bufPtr,"-texres",7)==0)
			_bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-type",5)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-mm",3)==0)
		{
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		}
		else
		if (strncmp(_bufPtr,"-o",2)==0) // texture coord translation
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
		}
		else
		if (strncmp(_bufPtr,"-s",2)==0) // texture coord scale
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
		}
		else
		if (strncmp(_bufPtr,"-t",2)==0)
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
		}
		// get next word
		_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
    }

    if (mapType != CMTLPipelineMetadata::EMP_COUNT)
    {
        std::string path = tmpbuf;
        std::replace(path.begin(), path.end(), '\\', '/');
        _currMaterial->maps[mapType] = std::move(path);
    }

    return _bufPtr;
}

std::pair<core::smart_refctd_ptr<ICPUSpecializedShader>, core::smart_refctd_ptr<ICPUSpecializedShader>> CGraphicsPipelineLoaderMTL::getShaders(bool _hasUV)
{
    auto vs = getDefaultAsset<ICPUSpecializedShader, IAsset::ET_SPECIALIZED_SHADER>(_hasUV ? VERT_SHADER_UV_CACHE_KEY : VERT_SHADER_NO_UV_CACHE_KEY, m_assetMgr);
    auto fs = getDefaultAsset<ICPUSpecializedShader, IAsset::ET_SPECIALIZED_SHADER>(_hasUV ? FRAG_SHADER_UV_CACHE_KEY : FRAG_SHADER_NO_UV_CACHE_KEY, m_assetMgr);

    return { std::move(vs), std::move(fs) };
}

auto CGraphicsPipelineLoaderMTL::loadImages(const char* _relDir, const SMtl& _mtl, SContext& _ctx) -> image_views_set_t
{
    images_set_t images;

    std::string relDir = _relDir;
    for (uint32_t i = 0u; i < images.size(); ++i)
    {
        SAssetLoadParams lp;
        if (_mtl.maps[i].size())
        {
            auto bundle = interm_getAssetInHierarchy(m_assetMgr, relDir+_mtl.maps[i], lp, _ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMAGE_HIERARCHYLEVELS_BELOW, _ctx.loaderOverride);
            if (!bundle.isEmpty())
                images[i] = core::smart_refctd_ptr_static_cast<ICPUImage>(bundle.getContents().first[0]);
        }
    }

    auto allCubemapFacesAreSameSizeAndFormat = [](const core::smart_refctd_ptr<ICPUImage>* _faces) {
        const VkExtent3D sz = (*_faces)->getCreationParameters().extent;
        const E_FORMAT fmt = (*_faces)->getCreationParameters().format;
        for (uint32_t i = 1u; i < 6u; ++i)
        {
            const auto& img = _faces[i];
            if (!img)
                continue;

            if (img->getCreationParameters().format != fmt)
                return false;
            const VkExtent3D sz_ = img->getCreationParameters().extent;
            if (sz.width != sz_.width || sz.height != sz_.height || sz.depth != sz_.depth)
                return false;
        }
        return true;
    };
    //make reflection cubemap
    if (images[CMTLPipelineMetadata::EMP_REFL_POSX])
    {
        assert(allCubemapFacesAreSameSizeAndFormat(images.data() + CMTLPipelineMetadata::EMP_REFL_POSX));

        size_t bufSz = 0ull;
        //assuming all cubemap layer images are same size and same format
        const size_t alignment = 1u<<core::findLSB(images[CMTLPipelineMetadata::EMP_REFL_POSX]->getRegions().begin()->bufferRowLength);
        core::vector<ICPUImage::SBufferCopy> regions_;
        regions_.reserve(6ull);
        for (uint32_t i = CMTLPipelineMetadata::EMP_REFL_POSX; i < CMTLPipelineMetadata::EMP_REFL_POSX + 6u; ++i)
        {
            assert(images[i]);
#ifndef _IRR_DEBUG
            if (images[i])
            {
#endif
                //assuming each image has just 1 region
                assert(images[i]->getRegions().length()==1ull);

                regions_.push_back(images[i]->getRegions().begin()[0]);
                regions_.back().bufferOffset = core::roundUp(regions_.back().bufferOffset, alignment);
                regions_.back().imageSubresource.baseArrayLayer = (i - CMTLPipelineMetadata::EMP_REFL_POSX);

                bufSz += images[i]->getImageDataSizeInBytes();
#ifndef _IRR_DEBUG
            }
#endif
        }
        auto imgDataBuf = core::make_smart_refctd_ptr<ICPUBuffer>(bufSz);
        for (uint32_t i = CMTLPipelineMetadata::EMP_REFL_POSX, j = 0u; i < CMTLPipelineMetadata::EMP_REFL_POSX + 6u; ++i)
        {
#ifndef _IRR_DEBUG
            if (images[i])
            {
#endif
                void* dst = reinterpret_cast<uint8_t*>(imgDataBuf->getPointer()) + regions_[j].bufferOffset;
                const void* src = reinterpret_cast<const uint8_t*>(images[i]->getBuffer()->getPointer()) + images[i]->getRegions().begin()[0].bufferOffset;
                const size_t sz = images[i]->getImageDataSizeInBytes();
                memcpy(dst, src, sz);

                ++j;
#ifndef _IRR_DEBUG
            }
#endif
        }

        //assuming all cubemap layer images are same size and same format
        ICPUImage::SCreationParams cubemapParams = images[CMTLPipelineMetadata::EMP_REFL_POSX]->getCreationParameters();
        cubemapParams.arrayLayers = 6u;
        cubemapParams.type = IImage::ET_2D;

        auto cubemap = ICPUImage::create(std::move(cubemapParams));
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(regions_);
        cubemap->setBufferAndRegions(std::move(imgDataBuf), regions);
        //new image goes to EMP_REFL_POSX index and other ones get nulled-out
        images[CMTLPipelineMetadata::EMP_REFL_POSX] = std::move(cubemap);
        for (uint32_t i = CMTLPipelineMetadata::EMP_REFL_POSX + 1u; i < CMTLPipelineMetadata::EMP_REFL_POSX + 6u; ++i)
        {
            images[i] = nullptr;
        }
    }

    image_views_set_t views;
    for (uint32_t i = 0u; i < views.size(); ++i)
    {
        if (!images[i])
            continue;

        const std::string viewCacheKey = _mtl.maps[i] + "?view";
        if (auto view = getDefaultAsset<ICPUImageView,IAsset::ET_IMAGE_VIEW>(viewCacheKey.c_str(), m_assetMgr))
        {
            views[i] = std::move(view);
            continue;
        }

        constexpr IImageView<ICPUImage>::E_TYPE viewType[2]{ IImageView<ICPUImage>::ET_2D, IImageView<ICPUImage>::ET_CUBE_MAP };
        constexpr uint32_t layerCount[2]{ 1u, 6u };

        const bool isCubemap = (i == CMTLPipelineMetadata::EMP_REFL_POSX);

        ICPUImageView::SCreationParams viewParams;
        viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
        viewParams.format = images[i]->getCreationParameters().format;
        viewParams.viewType = viewType[isCubemap];
        viewParams.subresourceRange.baseArrayLayer = 0u;
        viewParams.subresourceRange.layerCount = layerCount[isCubemap];
        viewParams.subresourceRange.baseMipLevel = 0u;
        viewParams.subresourceRange.levelCount = 1u;
        viewParams.image = std::move(images[i]);

        views[i] = ICPUImageView::create(std::move(viewParams));

        SAssetBundle bundle{views[i]};
        _ctx.loaderOverride->insertAssetIntoCache(bundle, viewCacheKey, _ctx.inner, _ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMAGEVIEW_HIERARCHYLEVELS_BELOW);
    }

    return views;
}

core::smart_refctd_ptr<ICPUDescriptorSet> CGraphicsPipelineLoaderMTL::makeDescSet(image_views_set_t&& _views, ICPUDescriptorSetLayout* _dsLayout)
{
    if (!_dsLayout)
        return nullptr;

    auto ds = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(
        core::smart_refctd_ptr<ICPUDescriptorSetLayout>(_dsLayout)
        );
    auto dummy2d = getDefaultAsset<ICPUImageView, IAsset::ET_IMAGE_VIEW>("irr/builtin/image_views/dummy2d", m_assetMgr);
    for (uint32_t i = 0u; i <= CMTLPipelineMetadata::EMP_REFL_POSX; ++i)
    {
        auto desc = ds->getDescriptors(i).begin();

        desc->desc = _views[i] ? std::move(_views[i]) : dummy2d;
        desc->image.imageLayout = EIL_UNDEFINED;
        desc->image.sampler = nullptr; //not needed, immutable (in DS layout) samplers are used
    }

    return ds;
}

auto CGraphicsPipelineLoaderMTL::readMaterials(io::IReadFile* _file) const -> core::vector<SMtl>
{
    std::string mtl;
    mtl.resize(_file->getSize());
    _file->read(mtl.data(), _file->getSize());

    const char* bufPtr = mtl.c_str();
    const char* const bufEnd = mtl.c_str()+mtl.size();

    constexpr static size_t WORD_BUFFER_LENGTH = 512ull;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

    auto readFloat = [&tmpbuf, &bufPtr, bufEnd] {
        float f = 0.f;

        bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
        sscanf(tmpbuf, "%f", &f);

        return f;
    };
    auto readRGB = [&readFloat] {
        core::vector3df_SIMD rgb(1.f);

        rgb.r = readFloat();
        rgb.g = readFloat();
        rgb.b = readFloat();

        return rgb;
    };

    core::vector<SMtl> materials;
    SMtl* currMaterial = nullptr;

    while (bufPtr != bufEnd)
    {
        copyWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
        if (currMaterial && (strncmp("map_", tmpbuf, 4u)==0 || strcmp("refl", tmpbuf)==0 || strcmp("norm", tmpbuf)==0 || strcmp("bump", tmpbuf)==0 || strcmp("disp", tmpbuf)==0))
        {
            readTexture(bufPtr, bufEnd, currMaterial, tmpbuf);
        }

        switch (*bufPtr)
        {
        case 'n': // newmtl
        {
            materials.push_back({});
            currMaterial = &materials.back();

            // extract new material's name
            bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);

            currMaterial->name = tmpbuf;
        }
        break;
        case 'a': // aniso, anisor
            if (currMaterial)
            {
                if (bufPtr[5] == 'r')
                    currMaterial->params.anisoRotation = readFloat();
                else
                    currMaterial->params.anisotropy = readFloat();
            }
        break;
        case 'i': // illum - illumination
            if (currMaterial)
            {
                bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
                currMaterial->params.extra |= (atol(tmpbuf)&0x0f);//illum values are in range [0;10]
            }
            break;
        case 'N':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 's': // Ns - shininess
                    currMaterial->params.shininess = readFloat();
                    break;
                case 'i': // Ni - refraction index
                    currMaterial->params.IoR = readFloat();
                    break;
                }
            }
            break;
        case 'K':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'd':		// Kd = diffuse
                    currMaterial->params.diffuse = readRGB();
                    break;
                case 's':		// Ks = specular
                    currMaterial->params.specular = readRGB();
                    break;
                case 'a':		// Ka = ambience
                    currMaterial->params.ambient = readRGB();
                    break;
                case 'e':		// Ke = emissive
                    currMaterial->params.emissive = readRGB();
                    break;
                }	// end switch(bufPtr[1])
            }	// end case 'K': if (currMaterial)...
            break;
        case 'P':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'r':
                    currMaterial->params.roughness = readFloat();
                    break;
                case 'm':
                    currMaterial->params.metallic = readFloat();
                    break;
                case 's':
                    currMaterial->params.sheen = readFloat();
                    break;
                case 'c':
                    switch (bufPtr[2])
                    {
                    case 'r':
                        currMaterial->params.clearcoatRoughness = readFloat();
                        break;
                    case 0:
                        currMaterial->params.clearcoatThickness = readFloat();
                        break;
                    }
                    break;
                }
            }
            break;
        case 'd': // d - transparency
            if (currMaterial)
                currMaterial->params.opacity = readFloat();
            break;
        case 'T':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'f':		// Tf - Transmitivity
                    currMaterial->params.transmissionFilter = readRGB();
                    sscanf(tmpbuf, "%s, %s: Detected Tf parameter, it won't be used in generated shader - fallback to alpha=0.5 instead", _file->getFileName().c_str(), currMaterial->name.c_str());
                    os::Printer::log(tmpbuf, ELL_WARNING);
                    break;
                case 'r':       // Tr, transparency = 1.0-d
                    currMaterial->params.opacity = (1.f - readFloat());
                    break;
                }
            }
            break;
        default: // comments or not recognised
            break;
        } // end switch(bufPtr[0])
        // go to next line
        bufPtr = goNextLine(bufPtr, bufEnd);
    }	// end while (bufPtr)

    return materials;
}
