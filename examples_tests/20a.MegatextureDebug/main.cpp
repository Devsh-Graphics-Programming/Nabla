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

STextureData getTextureData(const asset::ICPUImage* _img, asset::ICPUTexturePacker* _packer)
{
    const auto& extent = _img->getCreationParameters().extent;

    asset::IImage::SSubresourceRange subres;
    subres.baseMipLevel = 0u;
    // TODO compute mipmap and stuff
    subres.levelCount = core::findLSB(core::roundDownToPoT<uint32_t>(std::max(extent.width, extent.height))) + 1;

    auto pgTabCoords = _packer->pack(_img, subres, asset::ISampler::ETC_MIRROR, asset::ISampler::ETC_MIRROR);
    return _packer->offsetToTextureData(pgTabCoords, _img);
}

constexpr const char* GLSL_VT_TEXTURES = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
#ifndef _NO_UV
layout (set = 0, binding = 0) uniform usampler2D pgTabTex[3];
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
#define ADDR_LAYER_SHIFT 12
#define ADDR_Y_SHIFT 6
#define ADDR_X_MASK 0x3fu

#define TEXTURE_TILE_PER_DIMENSION 64
#define PAGE_SZ 128
#define PAGE_SZ_LOG2 7
#define TILE_PADDING 8

vec3 unpackPageID(in uint pageID)
{
	vec2 uv = vec2(float(pageID & ADDR_X_MASK), float((pageID>>ADDR_Y_SHIFT) & ADDR_X_MASK))*(PAGE_SZ+2*TILE_PADDING) + TILE_PADDING;
	uv /= vec2(textureSize(physPgTex[1],0).xy);
	return vec3(uv, float(pageID >> ADDR_LAYER_SHIFT));
    //return vec3(vec2(TILE_PADDING)/vec2(textureSize(physPgTex[1],0).xy), 0.0);
}

vec4 vTextureGrad_helper(in vec2 virtualUV, int LoD, in mat2 gradients, in ivec2 originalTextureSz)
{
    int clippedLoD = max(originalTextureSz.x,originalTextureSz.y)>>PAGE_SZ_LOG2;
    int maxLoD = clippedLoD!=0 ? findMSB(clippedLoD):0;
	int pgtabLoD = min(LoD,maxLoD);
	int tilesInLodLevel = textureSize(pgTabTex[1], pgtabLoD).x;
	ivec2 tileCoord = ivec2(virtualUV.xy*vec2(tilesInLodLevel));
	uvec2 pageID = texelFetch(pgTabTex[1],tileCoord,pgtabLoD).rg;
    ivec2 originalMipSize = ivec2(originalTextureSz.x>>LoD,originalTextureSz.y>>LoD);//is there element-wise bitshift? like vec>>n?
    int originalMipSize_maxDim = max(originalMipSize.x,originalMipSize.y);
	vec3 physicalUV = unpackPageID(originalMipSize_maxDim<=(PAGE_SZ/2) ? pageID.y : pageID.x); // unpack to normalized coord offset + Layer in physical texture (just bit operations) and multiples
    vec2 tileFractionalCoordinate = fract(virtualUV.xy*tilesInLodLevel);//i dont get this
    /*
    if (originalMipSize_maxDim <= PAGE_SZ/2)
    {
        //@devsh please check this
        int tmp = LoD-maxLoD;
        vec2 subtileSz = vec2(float(PAGE_SZ>>tmp));
        vec2 scale = vec2(originalMipSize)/subtileSz;
        //i dont get this
        tileFractionalCoordinate = tileFractionalCoordinate*scale + unpackUnorm2x16(tilePacking[1].offsets[tmp-1]); // mul by scale then offset
    }
    else //i dont get this
    */
        tileFractionalCoordinate = (tileFractionalCoordinate*float(PAGE_SZ)+vec2(TILE_PADDING))/float(PAGE_SZ+2*TILE_PADDING);
	physicalUV.xy += tileFractionalCoordinate;
	return textureGrad(physPgTex[1],physicalUV,gradients[0],gradients[1]);
}

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

vec2 unpackVirtualUV(in uvec2 texData)
{
	return unpackUnorm2x16(texData.x);
}
vec2 unpackSize(in uvec2 texData)
{
	return unpackUnorm2x16(texData.y);
}
vec4 textureVT(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 scale = unpackSize(_texData);
    vec2 virtualUV = unpackVirtualUV(_texData);
    virtualUV += scale*uv;
    return vTextureGrad(virtualUV, dUV, scale*float(PAGE_SZ)*vec2(textureSize(pgTabTex[1],0)));
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
	params.WindowSize = dimension2d<uint32_t>(1024, 1024);
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
    // load textures and pack them
    device->getFileSystem()->addFileArchive("../../media/sponza.zip");
    {
        asset::IAssetLoader::SAssetLoadParams lp;
        auto meshes_bundle = am->getAsset("sponza.obj", lp);
        assert(!meshes_bundle.isEmpty());
        auto mesh = meshes_bundle.getContents().first[0];
        auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());
        for (uint32_t i = 0u; i < mesh_raw->getMeshBufferCount(); ++i)
        {
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
                    texData = getTextureData(img.get(), texPackers[format2texPackerIndex(fmt)].get());
                    VTtexDataMap.insert({ img,texData });
                }
            }
        }
        am->removeAssetFromCache(meshes_bundle);
    }


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
        auto sampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);

        std::array<core::smart_refctd_ptr<asset::ICPUSampler>,ETP_COUNT> samplers;
        std::fill(samplers.begin(), samplers.end(), sampler);

        params.AnisotropicFilter = 0u;
        params.MaxFilter = asset::ISampler::ETF_NEAREST;
        params.MinFilter = asset::ISampler::ETF_NEAREST;
        params.MipmapMode = asset::ISampler::ESMM_NEAREST;
        auto samplerPgt = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);

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
    auto ds0 = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(core::smart_refctd_ptr(ds0layout));// intentionally not moving layout, let this pointer remain valid
    {
        // TODO: Move ICPUImageView creation to a ITexturePacker method
        auto pagetabs = ds0->getDescriptors(0u);
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            auto& pgtDesc = pagetabs.begin()[i];

            auto* img = texPackers[i]->getPageTable();

            asset::ICPUImageView::SCreationParams params;
            params.flags = static_cast<asset::IImageView<asset::ICPUImage>::E_CREATE_FLAGS>(0);
            params.format = img->getCreationParameters().format;
            params.subresourceRange.aspectMask = static_cast<asset::IImage::E_ASPECT_FLAGS>(0);
            params.subresourceRange.baseArrayLayer = 0u;
            params.subresourceRange.layerCount = img->getCreationParameters().arrayLayers;
            params.subresourceRange.baseMipLevel = 0u;
            params.subresourceRange.levelCount = img->getCreationParameters().mipLevels;
            params.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;//TODO change to ET_2D_ARRAY when pagetab is also layered
            params.image = core::smart_refctd_ptr<asset::ICPUImage>(img);

            pgtDesc.image.imageLayout = asset::EIL_UNDEFINED;
            pgtDesc.desc = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(params));
        }

        auto physAddrTexs = ds0->getDescriptors(1u);
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            auto& physTexDesc = physAddrTexs.begin()[i];

            auto* img = texPackers[i]->getPhysicalAddressTexture();

            asset::ICPUImageView::SCreationParams params;
            params.flags = static_cast<asset::IImageView<asset::ICPUImage>::E_CREATE_FLAGS>(0);
            params.format = img->getCreationParameters().format;
            params.subresourceRange.aspectMask = static_cast<asset::IImage::E_ASPECT_FLAGS>(0);
            params.subresourceRange.baseArrayLayer = 0u;
            params.subresourceRange.layerCount = img->getCreationParameters().arrayLayers;
            params.subresourceRange.baseMipLevel = 0u;
            params.subresourceRange.levelCount = img->getCreationParameters().mipLevels;
            params.viewType = asset::IImageView<asset::ICPUImage>::ET_2D_ARRAY;
            params.image = core::smart_refctd_ptr<asset::ICPUImage>(img);

            physTexDesc.image.imageLayout = asset::EIL_UNDEFINED;
            physTexDesc.desc = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(params));
        }
    }
    auto gpuds0 = driver->getGPUObjectsFromAssets(&ds0.get(),&ds0.get()+1)->front();

    core::smart_refctd_ptr<video::IGPUMeshBuffer> fsTriangleMeshBuffer;
    {
        asset::IAssetLoader::SAssetLoadParams lp;
        auto bundle = am->getAsset("../debugShader.frag", lp);
        assert(!bundle.isEmpty());
        auto pShader = &(*bundle.getContents().first);
        auto fragShader = driver->getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(pShader,pShader+1u);
        auto gpuds0layout = gpuds0->getLayout();
        auto pipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(const_cast<video::IGPUDescriptorSetLayout*>(gpuds0layout)));
        fsTriangleMeshBuffer = ext::FullScreenTriangle::createFullScreenTriangle(std::move(fragShader->front()),std::move(pipelineLayout),am,driver);
    }

	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,0,255,255) );

        auto pipeline = fsTriangleMeshBuffer->getPipeline();
        driver->bindGraphicsPipeline(pipeline);
        driver->bindDescriptorSets(video::EPBP_GRAPHICS,pipeline->getLayout(), 0u, 1u, &gpuds0.get(), nullptr);
        driver->drawMeshBuffer(fsTriangleMeshBuffer.get());

		driver->endScene();
	}

	return 0;
}