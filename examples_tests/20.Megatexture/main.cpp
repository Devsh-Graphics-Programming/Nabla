#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include <irr/asset/ITexturePacker.h>
#include <irr/asset/CMTLPipelineMetadata.h>
#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"

//#include "../../ext/ScreenShot/ScreenShot.h"
using namespace irr;
using namespace core;

using STextureData = asset::ITexturePacker::STextureData;

STextureData getTextureData(const asset::ICPUImage* _img, asset::ICPUTexturePacker* _packer, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
    const auto& extent = _img->getCreationParameters().extent;

    asset::IImage::SSubresourceRange subres;
    subres.baseMipLevel = 0u;
    subres.levelCount = core::findLSB(core::roundDownToPoT<uint32_t>(std::max(extent.width, extent.height))) + 1;

    uint8_t border[4]{};//unused anyway
    auto pgTabCoords = _packer->pack(_img, subres, asset::ISampler::ETC_MIRROR, asset::ISampler::ETC_MIRROR, _borderColor);
    return _packer->offsetToTextureData(pgTabCoords, _img);
}

constexpr const char* GLSL_VT_TEXTURES = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
#ifndef _NO_UV
layout (set = 0, binding = 0) uniform usampler2DArray pageTable[3];
layout (set = 0, binding = 1) uniform sampler2DArray physPgTex[3];
layout (set = 0, binding = 2) uniform TilePacking
{//TODO create this UBO
    uint offsets[9]; //unorm16 uv offsets in phys addr texture space
} tilePacking[3];
#endif
#define _IRR_FRAG_SET3_BINDINGS_DEFINED_
)";
constexpr const char* GLSL_PUSH_CONSTANTS_OVERRIDE =
R"(
layout (push_constant) uniform Block {
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    uvec2 map_Ka_data;
    uvec2 map_Kd_data;
    uvec2 map_Ks_data;
    uvec2 map_Ns_data;
    uvec2 map_d_data;
    uvec2 map_bump_data;
    float Ns;
    float d;
    float Ni;
    uint extra; //flags copied from MTL metadata
} PC;
#define _IRR_FRAG_PUSH_CONSTANTS_DEFINED_
)";
constexpr const char* GLSL_VT_FUNCTIONS =
R"(
#define ADDR_LAYER_SHIFT 12u
#define ADDR_Y_SHIFT 6u
#define ADDR_X_PREMASK 
#define ADDR_X_MASK uint((0x1u<<ADDR_Y_SHIFT)-1u)
#define ADDR_Y_MASK uint((0x1u<<(ADDR_LAYER_SHIFT-ADDR_Y_SHIFT))-1u)

#define PAGE_SZ 128
#define PAGE_SZ_LOG2 7
#define TILE_PADDING 8
#define PADDED_TILE_SIZE uint(PAGE_SZ+2*TILE_PADDING)

vec3 unpackPageID(in uint pageID)
{
	// this is optimal, don't touch
	uvec2 pageXY = uvec2(pageID,pageID>>ADDR_Y_SHIFT)&uvec2(ADDR_X_MASK,ADDR_Y_MASK);
	uvec2 pageOffset = pageXY*PADDED_TILE_SIZE;
	return vec3(vec2(pageOffset),pageID>>ADDR_LAYER_SHIFT);
}

vec4 vTextureGrad_helper(in vec3 virtualUV, int LoD, in mat2 gradients, in ivec2 originalTextureSz, in int clippedTextureLoD)
{
    uvec2 pageID = textureLod(pageTable[1],virtualUV,LoD).xy;

	// WANT: to get rid of this `textureSize` call
	float tilesInLodLevel = float(textureSize(pageTable[1],LoD).x);
	// TODO: rename to tileCoordinate if the dimensions will stay like this
	vec2 tileFractionalCoordinate = fract(virtualUV.xy*tilesInLodLevel);

	int levelInTail = max(LoD-clippedTextureLoD,0);
	tileFractionalCoordinate *= float(PAGE_SZ)*intBitsToFloat((127-levelInTail)<<23); // IEEE754 hack
	//tileFractionalCoordinate += packingOffsets[levelInTail];

	vec3 physicalUV = unpackPageID(levelInTail!=0 ? pageID.y:pageID.x);
	// add the in-tile coordinate
	physicalUV.xy += tileFractionalCoordinate;
	// scale from absolute coordinate to normalized (could actually use a denormalized sampler for this)
	physicalUV.xy /= vec2(textureSize(physPgTex[1],0).xy);

	return textureGrad(physPgTex[1],physicalUV,gradients[0],gradients[1]);
}

/*
float lengthSq(in vec2 v)
{
  return dot(v,v);
}
// textureGrad emulation
vec4 vTextureGrad(in vec2 virtualUV, in mat2 dOriginalUV, in vec2 originalTextureSize)
{
  // returns what would have been `textureGrad(originalTexture,gOriginalUV[0],gOriginalUV[1])
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap15.html#textures-normalized-operations
  const float kMaxAnisotropy = float(2*TILE_PADDING);
  const float kMaxAnisoLogOffset = log2(kMaxAnisotropy);
  // you can use an approx `log2` if you know one
  float p_x_2_log2 = log2(lengthSq(dOriginalUV[0]*originalTextureSize));
  float p_y_2_log2 = log2(lengthSq(dOriginalUV[1]*originalTextureSize));
  bool xIsMajor = p_x_2_log2>p_y_2_log2;
  float p_min_2_log2 = xIsMajor ? p_y_2_log2:p_x_2_log2;
  float p_max_2_log2 = xIsMajor ? p_x_2_log2:p_y_2_log2;
  float LoD = max(p_min_2_log2,p_max_2_log2-kMaxAnisoLogOffset);
  int LoD_high = int(LoD);
#ifdef DRIVER_HAS_GRADIENT_SCALING_ISSUES // don't use if you don't have to
  // normally when sampling an imageview with only 1 miplevel using `textureGrad` the gradient vectors can be scaled arbitrarily because we cannot select any other mip-map
  // however driver might want to try and get "clever" on us and decide that since `p_min` is huge, it wont bother with anisotropy
  float anisotropy = min(exp2((p_max_2_log2-p_min_2_log2)*0.5),kMaxAnisotropy);
  dOriginalUV[0] = normalize(dOriginalUV[0])*(1.0/float(MEGA_TEXTURE_SIZE));
  dOriginalUV[1] = normalize(dOriginalUV[1])*(1.0/float(MEGA_TEXTURE_SIZE));
  dOriginalUV[xIsMajor ? 0:1] *= anisotropy;
#endif
  ivec2 originalTexSz_i = ivec2(originalTextureSize);
  //return mix(vTextureGrad_helper(virtualUV,LoD_high,dOriginalUV,originalTexSz_i),vTextureGrad_helper(virtualUV,LoD_high+1,dOriginalUV,originalTexSz_i),LoD-float(LoD_high));
    return vTextureGrad_helper(virtualUV,0,dOriginalUV,originalTexSz_i);//testing purposes, until mips are present in physical tex pages
}
*/

//#define UNPACK_UNORM16_SANITY_CHECK
vec2 unpackUnorm2x16_(in uint u)
{
#ifdef UNPACK_UNORM16_SANITY_CHECK
    return vec2(float(u&0xffffu)/65535.0, float((u>>16)&0xffffu)/65535.0);
#else
    return unpackUnorm2x16(u);
#endif
}

vec2 unpackVirtualUV(in uvec2 texData)
{
	return unpackUnorm2x16_(texData.x);
}
vec2 unpackSize(in uvec2 texData)
{
	return unpackUnorm2x16_(texData.y);
}
vec4 textureVT(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 scale = unpackSize(_texData);
    vec2 virtualUV = unpackVirtualUV(_texData);
    //manual uv wrapping (repeat mode)
    virtualUV += 0.5*scale*mod(uv,vec2(1.0)); //why do i need factor of 0.5??????
    int whatever = 1000;//obv change later
    return vTextureGrad_helper(vec3(virtualUV,0.0), 0, mat2(0.0,0.0,0.0,0.0), ivec2(vec2(1.0)/scale*float(PAGE_SZ)), whatever);
}
)";
constexpr const char* GLSL_BSDF_COS_EVAL_OVERRIDE =
R"(
vec3 irr_bsdf_cos_eval(in irr_glsl_BSDFIsotropicParams params, in mat2 dUV)
{
    vec3 Kd;
#ifndef _NO_UV
    if ((PC.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Kd = textureVT(PC.map_Kd_data, UV, dUV).rgb;
    else
#endif
        Kd = PC.Kd;

    vec3 color = vec3(0.0);
    vec3 Ks;
    float Ns;

#ifndef _NO_UV
    if ((PC.extra&(map_Ks_MASK)) == (map_Ks_MASK))
        Ks = textureVT(PC.map_Ks_data, UV, dUV).rgb;
    else
#endif
        Ks = PC.Ks;
#ifndef _NO_UV
    if ((PC.extra&(map_Ns_MASK)) == (map_Ns_MASK))
        Ns = textureVT(PC.map_Ns_data, UV, dUV).x;
    else
#endif
        Ns = PC.Ns;

    vec3 Ni = vec3(PC.Ni);

    vec3 diff = irr_glsl_lambertian_cos_eval(params) * Kd * (1.0-irr_glsl_fresnel_dielectric(Ni,params.NdotL)) * (1.0-irr_glsl_fresnel_dielectric(Ni,params.NdotV));
    diff *= irr_glsl_diffuseFresnelCorrectionFactor(Ni, Ni*Ni);
    switch (PC.extra&ILLUM_MODEL_MASK)
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
#define _IRR_BSDF_COS_EVAL_DEFINED_
)";
constexpr const char* GLSL_COMPUTE_LIGHTING_OVERRIDE =
R"(
vec3 irr_computeLighting(out irr_glsl_ViewSurfaceInteraction out_interaction, in mat2 dUV)
{
    irr_glsl_ViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(vec3(0.0), ViewPos, Normal);

#ifndef _NO_UV
    if ((PC.extra&map_bump_MASK) == map_bump_MASK)
    {
        interaction.N = normalize(interaction.N);

        float h = textureVT(PC.map_bump_data, UV, dUV).x;

        vec2 dHdScreen = vec2(dFdx(h), dFdy(h));
        interaction.N = irr_glsl_perturbNormal_heightMap(interaction.N, interaction.V.dPosdScreen, dHdScreen);
    }
#endif
    irr_glsl_BSDFIsotropicParams params = irr_glsl_calcBSDFIsotropicParams(interaction, -ViewPos);

    vec3 Ka;
    switch ((PC.extra&ILLUM_MODEL_MASK))
    {
    case 0:
    {
#ifndef _NO_UV
    if ((PC.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Ka = textureVT(PC.map_Kd_data, UV, dUV).rgb;
    else
#endif
        Ka = PC.Kd;
    }
    break;
    default:
#define Ia 0.1
    {
#ifndef _NO_UV
    if ((PC.extra&(map_Ka_MASK)) == (map_Ka_MASK))
        Ka = textureVT(PC.map_Ka_data, UV, dUV).rgb;
    else
#endif
        Ka = PC.Ka;
    Ka *= Ia;
    }
#undef Ia
    break;
    }

    out_interaction = params.interaction;
#define Intensity 1000.0
    return Intensity*params.invlenL2*irr_bsdf_cos_eval(params,dUV) + Ka;
#undef Intensity
}
#define _IRR_COMPUTE_LIGHTING_DEFINED_
)";

constexpr const char* GLSL_FS_MAIN_OVERRIDE =
R"(
void main()
{
    mat2 dUV = mat2(dFdx(UV),dFdy(UV));

    irr_glsl_ViewSurfaceInteraction interaction;
    vec3 color = irr_computeLighting(interaction,dUV);

    float d = PC.d;

    //another illum model switch, required for illum=4,6,7,9 to compute alpha from fresnel (taken from opacity map or constant otherwise)
    switch (PC.extra&ILLUM_MODEL_MASK)
    {
    case 4:
    case 6:
    case 7:
    case 9:
    {
        float VdotN = dot(interaction.N, interaction.V.dir);
        d = irr_glsl_fresnel_dielectric(vec3(PC.Ni), VdotN).x;
    }
        break;
    default:
#ifndef _NO_UV
        if ((PC.extra&(map_d_MASK)) == (map_d_MASK))
        {
            d = textureVT(PC.map_d_data, UV, dUV).r;
            color *= d;
        }
#endif
        break;
    }

    OutColor = vec4(color, d);
}
#define _IRR_FRAG_MAIN_DEFINED_
)";

constexpr uint32_t TEX_OF_INTEREST_CNT = 6u;
#include "irr/irrpack.h"
struct SPushConstants
{
    //Ka
    core::vector3df_SIMD ambient;
    //Kd
    core::vector3df_SIMD diffuse;
    //Ks
    core::vector3df_SIMD specular;
    //Ke
    core::vector3df_SIMD emissive;
    uint64_t map_data[TEX_OF_INTEREST_CNT];
    //Ns, specular exponent in phong model
    float shininess = 32.f;
    //d
    float opacity = 1.f;
    //Ni, index of refraction
    float IoR = 1.6f;
    uint32_t extra;
} PACK_STRUCT;
#include "irr/irrunpack.h"
static_assert(sizeof(SPushConstants)<=asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "doesnt fit in push constants");

constexpr uint32_t texturesOfInterest[TEX_OF_INTEREST_CNT]{
    asset::CMTLPipelineMetadata::EMP_AMBIENT,
    asset::CMTLPipelineMetadata::EMP_DIFFUSE,
    asset::CMTLPipelineMetadata::EMP_SPECULAR,
    asset::CMTLPipelineMetadata::EMP_SHININESS,
    asset::CMTLPipelineMetadata::EMP_OPACITY,
    asset::CMTLPipelineMetadata::EMP_BUMP
};

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedFragShader(const asset::ICPUSpecializedShader* _fs)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    std::string glsl = reinterpret_cast<const char*>( unspec->getSPVorGLSL()->getPointer() );
    glsl.insert(glsl.find("#ifndef _IRR_FRAG_PUSH_CONSTANTS_DEFINED_"), GLSL_PUSH_CONSTANTS_OVERRIDE);
    glsl.insert(glsl.find("#if !defined(_IRR_FRAG_SET3_BINDINGS_DEFINED_)"), GLSL_VT_TEXTURES);
    glsl.insert(glsl.find("#ifndef _IRR_BSDF_COS_EVAL_DEFINED_"), GLSL_VT_FUNCTIONS);
    glsl.insert(glsl.find("#ifndef _IRR_BSDF_COS_EVAL_DEFINED_"), GLSL_BSDF_COS_EVAL_OVERRIDE);
    glsl.insert(glsl.find("#ifndef _IRR_COMPUTE_LIGHTING_DEFINED_"), GLSL_COMPUTE_LIGHTING_OVERRIDE);
    glsl.insert(glsl.find("#ifndef _IRR_FRAG_MAIN_DEFINED_"), GLSL_FS_MAIN_OVERRIDE);

    auto* f = fopen("fs.glsl","w");
    fwrite(glsl.c_str(), 1, glsl.size(), f);
    fclose(f);

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str());
    auto specinfo = _fs->getSpecializationInfo();//intentional copy
    auto fsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return fsNew;
}

constexpr uint32_t PAGETAB_SZ_LOG2 = 7u;
constexpr uint32_t PAGETAB_MIP_LEVELS = 8u;
constexpr uint32_t PAGE_SZ_LOG2 = 7u;
constexpr uint32_t TILES_PER_DIM_LOG2 = 6u;
constexpr uint32_t PHYS_ADDR_TEX_LAYERS = 3u;
constexpr uint32_t PAGE_PADDING = 8u;
enum E_TEX_PACKER
{
    ETP_8BIT,
    ETP_24BIT,
    ETP_32BIT,

    ETP_COUNT
};

E_TEX_PACKER format2texPackerIndex(asset::E_FORMAT _fmt)
{
    switch (asset::getFormatClass(_fmt))
    {
    case asset::EFC_8_BIT: return ETP_8BIT;
    case asset::EFC_24_BIT: return ETP_24BIT;
    case asset::EFC_32_BIT: return ETP_32BIT;
    default: return ETP_COUNT;
    }
}

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.


	//! disable mouse cursor, since camera will force it to the middle
	//! and we don't want a jittery cursor in the middle distracting us
	device->getCursorControl()->setVisible(false);

	//! Since our cursor will be enslaved, there will be no way to close the window
	//! So we listen for the "Q" key being pressed and exit the application
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();

    core::smart_refctd_ptr<asset::ICPUTexturePacker> texPackers[ETP_COUNT]{
        core::make_smart_refctd_ptr<asset::ICPUTexturePacker>(asset::EFC_8_BIT, asset::EF_R8_UNORM, PAGETAB_SZ_LOG2, PAGETAB_MIP_LEVELS, PAGE_SZ_LOG2, TILES_PER_DIM_LOG2, PHYS_ADDR_TEX_LAYERS, PAGE_PADDING),
        core::make_smart_refctd_ptr<asset::ICPUTexturePacker>(asset::EFC_24_BIT, asset::EF_R8G8B8_UNORM, PAGETAB_SZ_LOG2, PAGETAB_MIP_LEVELS, PAGE_SZ_LOG2, TILES_PER_DIM_LOG2, PHYS_ADDR_TEX_LAYERS, PAGE_PADDING),
        core::make_smart_refctd_ptr<asset::ICPUTexturePacker>(asset::EFC_32_BIT, asset::EF_R8G8B8A8_UNORM, PAGETAB_SZ_LOG2, PAGETAB_MIP_LEVELS, PAGE_SZ_LOG2, TILES_PER_DIM_LOG2, PHYS_ADDR_TEX_LAYERS, PAGE_PADDING),
    };
    //TODO most of sponza textures are 24bit format, but also need packers for 8bit and 32bit formats and a way to decide which VT textures to sample as well
    core::unordered_map<core::smart_refctd_ptr<asset::ICPUImage>, STextureData> VTtexDataMap;
    core::unordered_map<core::smart_refctd_ptr<asset::ICPUSpecializedShader>, core::smart_refctd_ptr<asset::ICPUSpecializedShader>> modifiedShaders;

    //TODO ds0 will most likely also get some UBO with data for MDI (instead of using push constants)
    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds0layout;
    {
        // TODO: Move sampler creation to a ITexturePacker method
        // also most basic and commonly used samplers should be built-in and cached in the IAssetManager
        asset::ICPUSampler::SParams params;
        params.AnisotropicFilter = 3u;
        params.BorderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_WHITE;
        params.CompareEnable = false;
        params.CompareFunc = asset::ISampler::ECO_NEVER;
        params.LodBias = 0.f;
        params.MaxLod = 10000.f;
        params.MinLod = 0.f;
        params.MaxFilter = asset::ISampler::ETF_LINEAR;
        params.MinFilter = asset::ISampler::ETF_LINEAR;
        //phys addr texture doesnt have mips anyway and page table is accessed with texelFetch only
        params.MipmapMode = asset::ISampler::ESMM_NEAREST;
        params.TextureWrapU = params.TextureWrapV = params.TextureWrapW = asset::ISampler::ETC_CLAMP_TO_EDGE;
        auto sampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params); //driver->createGPUSampler(params);

        std::array<core::smart_refctd_ptr<asset::ICPUSampler>,ETP_COUNT> samplers;
        std::fill(samplers.begin(), samplers.end(), sampler);

        params.AnisotropicFilter = 0u;
        params.MaxFilter = asset::ISampler::ETF_NEAREST;
        params.MinFilter = asset::ISampler::ETF_NEAREST;
        params.MipmapMode = asset::ISampler::ESMM_NEAREST;
        auto samplerPgt = core::make_smart_refctd_ptr<asset::ICPUSampler>(params); //driver->createGPUSampler(params);

        std::array<core::smart_refctd_ptr<asset::ICPUSampler>, ETP_COUNT> samplersPgt;
        std::fill(samplersPgt.begin(), samplersPgt.end(), samplerPgt);

        std::array<asset::ICPUDescriptorSetLayout::SBinding, 2> bindings;
        //page tables
        bindings[0].binding = 0u;
        bindings[0].count = ETP_COUNT;
        bindings[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
        bindings[0].samplers = samplersPgt.data();

        //physical addr textures
        bindings[1].binding = 1u;
        bindings[1].count = ETP_COUNT;
        bindings[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bindings[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
        bindings[1].samplers = samplers.data();

        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings.data(), bindings.data()+bindings.size());
    }
    core::smart_refctd_ptr<asset::ICPUPipelineLayout> pipelineLayout;
    {
        asset::SPushConstantRange pcrng;
        pcrng.offset = 0;
        pcrng.size = 128;
        pcrng.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;

        pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(&pcrng, &pcrng+1, core::smart_refctd_ptr(ds0layout), nullptr, nullptr, nullptr);
    }

    device->getFileSystem()->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("sponza.obj", lp);
    assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().first[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());
    //modifying push constants and default fragment shader for VT
    for (uint32_t i = 0u; i < mesh_raw->getMeshBufferCount(); ++i)
    {
        SPushConstants pushConsts;
        memset(pushConsts.map_data, 0xff, TEX_OF_INTEREST_CNT*sizeof(pushConsts.map_data[0]));
        pushConsts.extra = 0u;

        auto* mb = mesh_raw->getMeshBuffer(i);
        auto* ds = mb->getAttachedDescriptorSet();
        if (!ds)
            continue;
        for (uint32_t k = 0u; k < TEX_OF_INTEREST_CNT; ++k)
        {
            uint32_t j = texturesOfInterest[k];

            auto* view = static_cast<asset::ICPUImageView*>(ds->getDescriptors(j).begin()->desc.get());
            auto img = view->getCreationParameters().image;
            auto extent = img->getCreationParameters().extent;
            if (extent.width <= 2u || extent.height <= 2u)//dummy 2x2
                continue;
            STextureData texData;
            auto found = VTtexDataMap.find(img);
            if (found != VTtexDataMap.end())
                texData = found->second;
            else {
                const asset::E_FORMAT fmt = img->getCreationParameters().format;
                //TODO take wrapping into account while packing
                texData = getTextureData(img.get(), texPackers[format2texPackerIndex(fmt)].get(), asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK);
                VTtexDataMap.insert({img,texData});
            }

            static_assert(sizeof(texData)==sizeof(pushConsts.map_data[0]), "wrong reinterpret_cast");
            pushConsts.map_data[k] = reinterpret_cast<uint64_t*>(&texData)[0];
            uint32_t mapdata[2];
            memcpy(mapdata, &pushConsts.map_data[j], 8);
            printf("");
        }
        /*if (i == 0)
        {
            for (int k = 0; k < 6; ++k)
                reinterpret_cast<uint32_t*>(&pushConsts.map_data[k])[0] = reinterpret_cast<uint32_t*>(&pushConsts.map_data[k])[1] = 4123456789u;
        }*/
        auto* pipeline = mb->getPipeline();//TODO (?) might want to clone pipeline first, then modify and finally set into meshbuffer
        auto newPipeline = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(pipeline->clone(0u));//shallow copy
        //leave original ds1 layout since it's for UBO with matrices
        if (!pipelineLayout->getDescriptorSetLayout(1u))
            pipelineLayout->setDescriptorSetLayout(1u, core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(pipeline->getLayout()->getDescriptorSetLayout(1u)));
        newPipeline->setLayout(core::smart_refctd_ptr(pipelineLayout));
        {
            auto* fs = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX);
            auto found = modifiedShaders.find(core::smart_refctd_ptr<asset::ICPUSpecializedShader>(fs));
            core::smart_refctd_ptr<asset::ICPUSpecializedShader> newfs;
            if (found != modifiedShaders.end())
                newfs = found->second;
            else {
                newfs = createModifiedFragShader(fs);
                modifiedShaders.insert({core::smart_refctd_ptr<asset::ICPUSpecializedShader>(fs),newfs});
            }
            newPipeline->setShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, newfs.get());
        }
        auto* metadata = static_cast<asset::CMTLPipelineMetadata*>( pipeline->getMetadata() );
        am->setAssetMetadata(newPipeline.get(), core::smart_refctd_ptr<asset::IAssetMetadata>(metadata));
        //copy texture presence flags
        pushConsts.extra = metadata->getMaterialParams().extra;
        pushConsts.ambient = metadata->getMaterialParams().ambient;
        pushConsts.diffuse = metadata->getMaterialParams().diffuse;
        pushConsts.emissive = metadata->getMaterialParams().emissive;
        pushConsts.specular = metadata->getMaterialParams().specular;
        pushConsts.IoR = metadata->getMaterialParams().IoR;
        pushConsts.opacity = metadata->getMaterialParams().opacity;
        pushConsts.shininess = metadata->getMaterialParams().shininess;
        memcpy(mb->getPushConstantsDataPtr(), &pushConsts, sizeof(pushConsts));

        //we dont want this DS to be converted into GPU DS, so set to nullptr
        //dont worry about deletion of textures (invalidation of pointers), they're grabbed in VTtexDataMap
        mb->setAttachedDescriptorSet(nullptr);

        //set new pipeline (with overriden FS and layout)
        mb->setPipeline(std::move(newPipeline));
        //optionally adjust push constant ranges, but at worst it'll just be specified too much because MTL uses all 128 bytes
    }

    core::smart_refctd_ptr<video::IGPUTexturePacker> gpuTexPackers[ETP_COUNT];
    for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        gpuTexPackers[i] = core::make_smart_refctd_ptr<video::IGPUTexturePacker>(driver, texPackers[i].get());

    auto gpuds0layout = driver->getGPUObjectsFromAssets(&ds0layout.get(), &ds0layout.get()+1)->front();
    auto gpuds0 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuds0layout));//intentionally not moving layout
    {
        std::array<video::IGPUDescriptorSet::SWriteDescriptorSet, 2> writes;
        //page table
        video::IGPUDescriptorSet::SDescriptorInfo info1[ETP_COUNT];
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            info1[i].image.imageLayout = asset::EIL_UNDEFINED;
            info1[i].desc = gpuTexPackers[i]->createPageTableView(driver);
        }
        writes[0].binding = 0u;
        writes[0].arrayElement = 0u;
        writes[0].count = ETP_COUNT;
        writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
        writes[0].dstSet = gpuds0.get();
        writes[0].info = info1;
        //phys addr tex
        video::IGPUDescriptorSet::SDescriptorInfo info2[ETP_COUNT];
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            info2[i].image.imageLayout = asset::EIL_UNDEFINED;
            info2[i].desc = gpuTexPackers[i]->createPhysicalAddressTextureView(driver);
        }
        writes[1].binding = 1u;
        writes[1].arrayElement = 0u;
        writes[1].count = ETP_COUNT;
        writes[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
        writes[1].dstSet = gpuds0.get();
        writes[1].info = info2;

        driver->updateDescriptorSets(writes.size(), writes.data(), 0u, nullptr);
    }

    //we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
    //so we can create just one DS
    
    asset::ICPUDescriptorSetLayout* ds1layout = mesh_raw->getMeshBuffer(0u)->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
        {
            ds1UboBinding = bnd.binding;
            break;
        }

    size_t neededDS1UBOsz = 0ull;
    {
        auto pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(mesh_raw->getMeshBuffer(0u)->getPipeline()->getMetadata());
        for (const auto& shdrIn : pipelineMetadata->getCommonRequiredInputs())
            if (shdrIn.descriptorSection.type==asset::IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
    }

    auto gpuds1layout = driver->getGPUObjectsFromAssets(&ds1layout, &ds1layout+1)->front();

    auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(neededDS1UBOsz);
    auto gpuds1 = driver->createGPUDescriptorSet(std::move(gpuds1layout));
    {
        video::IGPUDescriptorSet::SWriteDescriptorSet write;
        write.dstSet = gpuds1.get();
        write.binding = ds1UboBinding;
        write.count = 1u;
        write.arrayElement = 0u;
        write.descriptorType = asset::EDT_UNIFORM_BUFFER;
        video::IGPUDescriptorSet::SDescriptorInfo info;
        {
            info.desc = gpuubo;
            info.buffer.offset = 0ull;
            info.buffer.size = neededDS1UBOsz;
        }
        write.info = &info;
        driver->updateDescriptorSets(1u, &write, 0u, nullptr);
    }

    auto gpumesh = driver->getGPUObjectsFromAssets(&mesh_raw, &mesh_raw+1)->front();

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.5f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(1000.0f);

    smgr->setActiveCamera(camera);

	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

        /*
        driver->bindGraphicsPipeline(fs_pipeline.get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, fs_pipelineLayout.get(), 0u, 1u, &gpuds0.get(), nullptr);

        driver->drawMeshBuffer(fs_meshbuf.get());
        */

        core::vector<uint8_t> uboData(gpuubo->getSize());
        auto pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(mesh_raw->getMeshBuffer(0u)->getPipeline()->getMetadata());
        for (const auto& shdrIn : pipelineMetadata->getCommonRequiredInputs())
        {
            if (shdrIn.descriptorSection.type==asset::IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                case asset::IPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                {
                    core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
                    memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                case asset::IPipelineMetadata::ECSI_WORLD_VIEW:
                {
                    core::matrix3x4SIMD MV = camera->getViewMatrix();
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                case asset::IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                {
                    core::matrix3x4SIMD MV = camera->getViewMatrix();
                    memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                }
            }
        }       
        driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

        for (uint32_t i = 0u; i < gpumesh->getMeshBufferCount(); ++i)
        {
            video::IGPUMeshBuffer* gpumb = gpumesh->getMeshBuffer(i);
            const video::IGPURenderpassIndependentPipeline* pipeline = gpumb->getPipeline();

            driver->bindGraphicsPipeline(pipeline);
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 1u, &gpuds0.get(), nullptr);
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 1u, 1u, &gpuds1.get(), nullptr);
            //const video::IGPUDescriptorSet* gpuds3_ptr = gpumb->getAttachedDescriptorSet();
            //if (gpuds3_ptr)
            //    driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            driver->pushConstants(pipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpumb->MAX_PUSH_CONSTANT_BYTESIZE, gpumb->getPushConstantsDataPtr());

            driver->drawMeshBuffer(gpumb);
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Meshloaders Demo - IrrlichtBAW Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		//ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	return 0;
}