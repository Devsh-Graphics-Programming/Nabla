#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include <irr/asset/ITexturePacker.h>
#include <irr/video/IGPUVirtualTexture.h>
#include <irr/asset/CMTLPipelineMetadata.h>
#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include <irr/asset/filters/CMipMapGenerationImageFilter.h>

//#include "../../ext/ScreenShot/ScreenShot.h"
using namespace irr;
using namespace core;

using STextureData = asset::ICPUVirtualTexture::STextureData;

STextureData getTextureData(const asset::ICPUImage* _img, asset::ICPUVirtualTexture* _vt, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
    const auto& extent = _img->getCreationParameters().extent;

    asset::IImage::SSubresourceRange subres;
    subres.baseMipLevel = 0u;
    subres.levelCount = core::findLSB(core::roundDownToPoT<uint32_t>(std::max(extent.width, extent.height))) + 1;

    uint8_t border[4]{};//unused anyway
    return _vt->pack(_img, subres, _uwrap, _vwrap, _borderColor);
}

constexpr const char* GLSL_VT_TEXTURES = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
#ifndef _NO_UV
#define VT_COUNT 3
layout (set = 0, binding = 0) uniform usampler2DArray pageTable;
layout (set = 0, binding = 1) uniform sampler2DArray physPgTex[VT_COUNT];
layout (set = 0, binding = 2, std430) readonly restrict buffer LUT
{
    uint lut[VT_COUNT];
} layer2pid;

layout (set = 2, binding = 0, std430) readonly restrict buffer PrecomputedStuffUBO
{
    uint pgtab_sz_log2[VT_COUNT];
    float phys_pg_tex_sz_rcp[VT_COUNT];
    float vtex_sz_rcp[VT_COUNT];
} precomputed;
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

uint getPhysicalIDForLayer(in uint layer)
{
    return layer2pid.lut[layer];
}

uint getPgTabSzLog2(in uint _formatID)
{
    return precomputed.pgtab_sz_log2[_formatID];
}
float getPhysPgTexSzRcp(in uint _formatID)
{
    return precomputed.phys_pg_tex_sz_rcp[_formatID];
}
float getVTexSzRcp(in uint _formatID)
{
    return precomputed.vtex_sz_rcp[_formatID];
}

#include <irr/builtin/glsl/virtual_texturing/functions.glsl/6/6/7/8/pageTable/physPgTex/getPgTabSzLog2/getPhysPgTexSzRcp/getVTexSzRcp/getPhysicalIDForLayer>
)";
constexpr const char* GLSL_BSDF_COS_EVAL_OVERRIDE =
R"(
vec3 irr_bsdf_cos_eval(in irr_glsl_BSDFIsotropicParams params, in mat2 dUV)
{
    vec3 Kd;
#ifndef _NO_UV
    if ((PC.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Kd = irr_glsl_textureVT(PC.map_Kd_data, UV, dUV).rgb;
    else
#endif
        Kd = PC.Kd;

    vec3 color = vec3(0.0);
    vec3 Ks;
    float Ns;

#ifndef _NO_UV
    if ((PC.extra&(map_Ks_MASK)) == (map_Ks_MASK))
        Ks = irr_glsl_textureVT(PC.map_Ks_data, UV, dUV).rgb;
    else
#endif
        Ks = PC.Ks;
#ifndef _NO_UV
    if ((PC.extra&(map_Ns_MASK)) == (map_Ns_MASK))
        Ns = irr_glsl_textureVT(PC.map_Ns_data, UV, dUV).x;
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

        float h = irr_glsl_textureVT(PC.map_bump_data, UV, dUV).x;

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
        Ka = irr_glsl_textureVT(PC.map_Kd_data, UV, dUV).rgb;
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
        Ka = irr_glsl_textureVT(PC.map_Ka_data, UV, dUV).rgb;
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
//#define COLOR_IS_DIFFUSE_TEX
#ifndef COLOR_IS_DIFFUSE_TEX
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
            d = irr_glsl_textureVT(PC.map_d_data, UV, dUV).r;
            color *= d;
        }
#endif
        break;
    }
#else//!COLOR_IS_DIFFUSE_TEX
    vec3 color = vec3(0.0);
    float d = 1.0;
#ifndef _NO_UV
    if ((PC.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        color = irr_glsl_textureVT(PC.map_Kd_data, UV, dUV).rgb;
    else
#endif
        color = PC.Kd;
#endif//!COLOR_IS_DIFFUSE_TEX
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
    size_t firstNewlineAfterVersion = glsl.find("\n",glsl.find("#version "));
    //glsl.insert(firstNewlineAfterVersion, GLSL_NONUNIFORM_EXT_OVERRIDE);
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

constexpr uint32_t PGTAB_SZ_LOG2 = 7u;
constexpr uint32_t PGTAB_LAYERS_PER_FORMAT = 1u;
constexpr uint32_t PGTAB_LAYERS = 3u;
constexpr uint32_t PAGE_SZ_LOG2 = 7u;
constexpr uint32_t TILES_PER_DIM_LOG2 = 6u;
constexpr uint32_t PHYS_ADDR_TEX_LAYERS = 3u;
constexpr uint32_t PAGE_PADDING = 8u;
constexpr uint32_t MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u; //4096
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

core::smart_refctd_ptr<asset::ICPUImage> createPoTPaddedSquareImageWithMipLevels(asset::ICPUImage* _img, asset::ISampler::E_TEXTURE_CLAMP _wrapu, asset::ISampler::E_TEXTURE_CLAMP _wrapv)
{
    const auto& params = _img->getCreationParameters();
    const uint32_t paddedExtent = core::roundUpToPoT(std::max(params.extent.width,params.extent.height));

    //create PoT and square image with regions for all mips
    asset::ICPUImage::SCreationParams paddedParams = params;
    paddedParams.extent = {paddedExtent,paddedExtent,1u};
    //in case of original extent being non-PoT, padding it to PoT gives us one extra not needed mip level (making sure to not cumpute it)
    paddedParams.mipLevels = core::findLSB(paddedExtent) + (core::isPoT(std::max(params.extent.width,params.extent.height)) ? 1 : 0);
    auto paddedImg = asset::ICPUImage::create(std::move(paddedParams));
    {
        const uint32_t texelBytesize = asset::getTexelOrBlockBytesize(params.format);

        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(paddedImg->getCreationParameters().mipLevels);
        uint32_t bufoffset = 0u;
        for (uint32_t i = 0u; i < regions->size(); ++i)
        {
            auto& region = (*regions)[i];
            region.bufferImageHeight = 0u;
            region.bufferOffset = bufoffset;
            region.bufferRowLength = paddedExtent>>i;
            region.imageExtent = {paddedExtent>>i,paddedExtent>>i,1u};
            region.imageOffset = {0u,0u,0u};
            region.imageSubresource.baseArrayLayer = 0u;
            region.imageSubresource.layerCount = 1u;
            region.imageSubresource.mipLevel = i;

            bufoffset += texelBytesize*region.imageExtent.width*region.imageExtent.height;
        }
        auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufoffset);
        paddedImg->setBufferAndRegions(std::move(buf), regions);
    }

    //copy mip 0 to new image while filling padding according to wrapping modes
    asset::CPaddedCopyImageFilter::state_type copy;
    copy.axisWraps[0] = _wrapu;
    copy.axisWraps[1] = _wrapv;
    copy.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
    copy.borderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK;
    copy.extent = params.extent;
    copy.layerCount = 1u;
    copy.inMipLevel = 0u;
    copy.inOffset = {0u,0u,0u};
    copy.inBaseLayer = 0u;
    copy.outOffset = {0u,0u,0u};
    copy.outBaseLayer = 0u;
    copy.outMipLevel = 0u;
    copy.paddedExtent = {paddedExtent,paddedExtent,1u};
    copy.relativeOffset = {0u,0u,0u};
    copy.inImage = _img;
    copy.outImage = paddedImg.get();

    asset::CPaddedCopyImageFilter::execute(&copy);

    using mip_gen_filter_t = asset::CMipMapGenerationImageFilter<asset::CBoxImageFilterKernel,asset::CBoxImageFilterKernel>;
    //generate all mip levels
    {
        mip_gen_filter_t::state_type genmips;
        genmips.baseLayer = 0u;
        genmips.layerCount = 1u;
        genmips.startMipLevel = 1u;
        genmips.endMipLevel = paddedImg->getCreationParameters().mipLevels;
        genmips.inOutImage = paddedImg.get();
        genmips.scratchMemoryByteSize = mip_gen_filter_t::getRequiredScratchByteSize(&genmips);
        genmips.scratchMemory = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(genmips.scratchMemoryByteSize,_IRR_SIMD_ALIGNMENT));
        mip_gen_filter_t::execute(&genmips);
        _IRR_ALIGNED_FREE(genmips.scratchMemory);
    }

    //bring back original extent
    {
        auto paddedRegions = paddedImg->getRegions();
        auto buf = core::smart_refctd_ptr<asset::ICPUBuffer>( paddedImg->getBuffer() );
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(paddedRegions.size());
        memcpy(regions->data(), paddedRegions.begin(), sizeof(asset::IImage::SBufferCopy)*regions->size());
        auto originalExtent = _img->getCreationParameters().extent;
        for (uint32_t i = 0u; i < regions->size(); ++i)
        {
            auto& region = (*regions)[i];
            region.imageExtent = {std::max(originalExtent.width>>i,1u),std::max(originalExtent.height>>i,1u),1u};
        }

        auto newParams = paddedImg->getCreationParameters();
        newParams.extent = originalExtent;
        paddedImg = asset::ICPUImage::create(std::move(newParams));
        paddedImg->setBufferAndRegions(std::move(buf), regions);
    }

    return paddedImg;
}

class EventReceiver : public irr::IEventReceiver
{
    _IRR_STATIC_INLINE_CONSTEXPR int32_t MAX_LOD = 8;
	public:
		bool OnEvent(const irr::SEvent& event)
		{
			if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
			{
				switch (event.KeyInput.Key)
				{
					case irr::KEY_KEY_Q: // switch wire frame mode
						running = false;
						return true;
                    case KEY_KEY_Z:
                        if (LoD>0)
                            --LoD;
                        return true;
                    case KEY_KEY_X:
                        if (LoD<MAX_LOD)
                            ++LoD;
                        return true;
					default:
						break;
				}
			}

			return false;
		}

		inline bool keepOpen() const { return running; }
        const int32_t& getLoD() const { return LoD; }

	private:
		bool running = true;
        int32_t LoD = 0;
};

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
	EventReceiver receiver;
	device->setEventReceiver(&receiver);

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();

    std::array<asset::ICPUVirtualTexture::ICPUVTResidentStorage::SCreationParams,3> storage;
    storage[0].formatClass = asset::EFC_8_BIT;
    storage[0].layerCount = 3u;
    storage[0].tilesPerDim_log2 = TILES_PER_DIM_LOG2;
    storage[0].formatCount = 1u;
    asset::E_FORMAT fmt1[1]{ asset::EF_R8_UNORM };
    storage[0].formats = fmt1;
    storage[1].formatClass = asset::EFC_24_BIT;
    storage[1].layerCount = 3u;
    storage[1].tilesPerDim_log2 = TILES_PER_DIM_LOG2;
    storage[1].formatCount = 1u;
    asset::E_FORMAT fmt2[1]{ asset::EF_R8G8B8_SRGB };
    storage[1].formats = fmt2;
    storage[2].formatClass = asset::EFC_32_BIT;
    storage[2].layerCount = 3u;
    storage[2].tilesPerDim_log2 = TILES_PER_DIM_LOG2;
    storage[2].formatCount = 1u;
    asset::E_FORMAT fmt3[1]{ asset::EF_R8G8B8A8_SRGB };
    storage[2].formats = fmt3;
    auto vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>(storage.data(), storage.size(), PGTAB_SZ_LOG2, PGTAB_LAYERS, PAGE_SZ_LOG2, PAGE_PADDING, MAX_ALLOCATABLE_TEX_SZ_LOG2);

    //auto pagetable = asset::ICPUTexturePacker::createPageTable(PGTAB_SZ_LOG2, PGTAB_LAYERS_PER_FORMAT*ETP_COUNT, PAGE_SZ_LOG2, MAX_ALLOCATABLE_TEX_SZ_LOG2);

    /*
    core::smart_refctd_ptr<asset::ICPUTexturePacker> texPackers[ETP_COUNT];
    asset::IImage::SSubresourceRange subresRange;
    subresRange.layerCount = PGTAB_LAYERS_PER_FORMAT;
    subresRange.baseArrayLayer = 0u*PGTAB_LAYERS_PER_FORMAT;
    texPackers[0] = core::make_smart_refctd_ptr<asset::ICPUTexturePacker>(asset::EF_R8_UNORM, core::smart_refctd_ptr(pagetable), subresRange, PAGE_SZ_LOG2, TILES_PER_DIM_LOG2, PHYS_ADDR_TEX_LAYERS, PAGE_PADDING);
    subresRange.baseArrayLayer = 1u*PGTAB_LAYERS_PER_FORMAT;
    texPackers[1] = core::make_smart_refctd_ptr<asset::ICPUTexturePacker>(asset::EF_R8G8B8_UNORM, core::smart_refctd_ptr(pagetable), subresRange, PAGE_SZ_LOG2, TILES_PER_DIM_LOG2, PHYS_ADDR_TEX_LAYERS, PAGE_PADDING);
    subresRange.baseArrayLayer = 2u*PGTAB_LAYERS_PER_FORMAT;
    texPackers[2] = core::make_smart_refctd_ptr<asset::ICPUTexturePacker>(asset::EF_R8G8B8A8_UNORM, core::smart_refctd_ptr(pagetable), subresRange, PAGE_SZ_LOG2, TILES_PER_DIM_LOG2, PHYS_ADDR_TEX_LAYERS, PAGE_PADDING);
    */
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
        auto sampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);

        std::array<core::smart_refctd_ptr<asset::ICPUSampler>,ETP_COUNT> samplers;
        std::fill(samplers.begin(), samplers.end(), sampler);

        params.AnisotropicFilter = 0u;
        params.MaxFilter = asset::ISampler::ETF_NEAREST;
        params.MinFilter = asset::ISampler::ETF_NEAREST;
        params.MipmapMode = asset::ISampler::ESMM_NEAREST;
        auto samplerPgt = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);

        std::array<asset::ICPUDescriptorSetLayout::SBinding, 3> bindings;
        //page tables
        bindings[0].binding = 0u;
        bindings[0].count = 1u;
        bindings[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
        bindings[0].samplers = &samplerPgt;

        //physical addr textures
        bindings[1].binding = 1u;
        bindings[1].count = ETP_COUNT;
        bindings[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bindings[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
        bindings[1].samplers = samplers.data();

        //pgtab layer -> physical page texture LUT SSBO
        bindings[2].binding = 2u;
        bindings[2].count = 1u;
        bindings[2].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bindings[2].type = asset::EDT_STORAGE_BUFFER;
        bindings[2].samplers = nullptr;

        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings.data(), bindings.data()+bindings.size());
    }
    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds2layout;
    {
        asset::ICPUDescriptorSetLayout::SBinding bnd;
        bnd.binding = 0u;
        bnd.count = 1u;
        bnd.samplers = nullptr;
        bnd.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bnd.type = asset::EDT_STORAGE_BUFFER;
        ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&bnd,&bnd+1);
    }

    core::smart_refctd_ptr<asset::ICPUPipelineLayout> pipelineLayout;
    {
        asset::SPushConstantRange pcrng;
        pcrng.offset = 0;
        pcrng.size = 128;
        pcrng.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;

        pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(&pcrng, &pcrng+1, core::smart_refctd_ptr(ds0layout), nullptr, core::smart_refctd_ptr(ds2layout), nullptr);
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
            auto* smplr = ds->getLayout()->getBindings().begin()[j].samplers[0].get();
            const auto uwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(smplr->getParams().TextureWrapU);
            const auto vwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(smplr->getParams().TextureWrapV);
            const auto borderColor = static_cast<asset::ISampler::E_TEXTURE_BORDER_COLOR>(smplr->getParams().BorderColor);
            auto img = view->getCreationParameters().image;
            auto extent = img->getCreationParameters().extent;
            if (extent.width <= 2u || extent.height <= 2u)//dummy 2x2
                continue;
            STextureData texData;
            auto found = VTtexDataMap.find(img);
            if (found != VTtexDataMap.end())
                texData = found->second;
            else {
                auto imgToPack = createPoTPaddedSquareImageWithMipLevels(img.get(), uwrap, vwrap);
                const asset::E_FORMAT fmt = imgToPack->getCreationParameters().format;
                texData = getTextureData(imgToPack.get(), vt.get(), uwrap, vwrap, borderColor);
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
    //default cpu2gpu shouldnt generate extra mips for integer format textures
    auto gpuvt = core::make_smart_refctd_ptr<video::IGPUVirtualTexture>(driver, vt.get());
    //auto gpuPagetable = driver->getGPUObjectsFromAssets(&pagetable.get(), &pagetable.get()+1)->front();
    //core::smart_refctd_ptr<video::IGPUTexturePacker> gpuTexPackers[ETP_COUNT];
    //for (uint32_t i = 0u; i < ETP_COUNT; ++i)
    //    gpuTexPackers[i] = core::make_smart_refctd_ptr<video::IGPUTexturePacker>(driver, texPackers[i].get(), core::smart_refctd_ptr(gpuPagetable));

    auto gpuds0layout = driver->getGPUObjectsFromAssets(&ds0layout.get(), &ds0layout.get()+1)->front();
    auto gpuds0 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuds0layout));//intentionally not moving layout
    {
        std::array<video::IGPUDescriptorSet::SWriteDescriptorSet, 3> writes;
        //page table
        video::IGPUDescriptorSet::SDescriptorInfo info0[1];
        //info0->desc = gpuTexPackers[0]->createPageTableView(driver);//doesnt matter which gpuTexPacker i use for this
        info0->desc = gpuvt->createPageTableView();
        info0->image.imageLayout = asset::EIL_UNDEFINED;
        writes[0].binding = 0u;
        writes[0].arrayElement = 0u;
        writes[0].count = 1u;
        writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
        writes[0].dstSet = gpuds0.get();
        writes[0].info = info0;
        //phys addr tex
        video::IGPUDescriptorSet::SDescriptorInfo info1[ETP_COUNT];
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            info1[i].image.imageLayout = asset::EIL_UNDEFINED;
            //info1[i].desc = gpuTexPackers[i]->createPhysicalAddressTextureView(driver);
            info1[i].desc = gpuvt->getFloatViews().begin()[i];
        }
        writes[1].binding = 1u;
        writes[1].arrayElement = 0u;
        writes[1].count = ETP_COUNT;
        writes[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
        writes[1].dstSet = gpuds0.get();
        writes[1].info = info1;
        //LUT SSBO
        video::IGPUDescriptorSet::SDescriptorInfo info2[1];
        {
            const size_t sz = gpuvt->getLayerToViewIndexMapping().size();
            info2->desc = driver->createFilledDeviceLocalGPUBufferOnDedMem(sz*sizeof(uint32_t), gpuvt->getLayerToViewIndexMapping().begin());
            info2->buffer.offset = 0u;
            info2->buffer.size = sz*sizeof(uint32_t);
        }
        writes[2].binding = 2u;
        writes[2].arrayElement = 0u;
        writes[2].count = 1u;
        writes[2].descriptorType = asset::EDT_STORAGE_BUFFER;
        writes[2].dstSet = gpuds0.get();
        writes[2].info = info2;

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

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpu_ds2layout = driver->getGPUObjectsFromAssets(&ds2layout.get(),&ds2layout.get()+1)->front();
    auto gpu_ds2 = driver->createGPUDescriptorSet(std::move(gpu_ds2layout));
    core::smart_refctd_ptr<video::IGPUBuffer> gpuPrecomputedStuffSsbo;
    {
        constexpr uint32_t ENTRY_SZ = sizeof(uint32_t);
        constexpr uint32_t ENTRY_ALIGNMENT = sizeof(uint32_t);//change to 16 if std140
        constexpr uint32_t PRECOMPUTED_STUFF_COUNT = 3u;//pgtab_sz_log2, phys_pg_tex_sz_rcp, vtex_sz_rcp
        constexpr uint32_t PRECOMPUTED_STUFF_SZ = ETP_COUNT*ENTRY_ALIGNMENT;
        constexpr uint32_t UBO_SZ = PRECOMPUTED_STUFF_COUNT*PRECOMPUTED_STUFF_SZ;
        uint8_t ubodata[UBO_SZ]{};
        uint32_t* ptr = reinterpret_cast<uint32_t*>(ubodata);
        uint32_t offset = 0u;
        //pgtab_sz_log2
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
            ptr[i*ENTRY_ALIGNMENT/ENTRY_SZ] = core::findMSB(gpuvt->getPageTable()->getCreationParameters().extent.width);
        ptr += PRECOMPUTED_STUFF_SZ/ENTRY_SZ;
        //phys_pg_tex_sz_rcp
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            const auto& storageImg = gpuvt->getFloatViews().begin()[i]->getCreationParameters().image;
            const double f = 1.0 / static_cast<double>(storageImg->getCreationParameters().extent.width);
            reinterpret_cast<float*>(ptr)[i*ENTRY_ALIGNMENT/ENTRY_SZ] = f;
        }
        ptr += PRECOMPUTED_STUFF_SZ/ENTRY_SZ;
        //vtex_sz_rcp
        for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        {
            double f = 1.0;
            f /= static_cast<double>(gpuvt->getPageTable()->getCreationParameters().extent.width);
            f /= static_cast<double>(gpuvt->getPageExtent());
            reinterpret_cast<float*>(ptr)[i*ENTRY_ALIGNMENT/ENTRY_SZ] = f;
        }
        gpuPrecomputedStuffSsbo = driver->createFilledDeviceLocalGPUBufferOnDedMem(UBO_SZ,ubodata);

        {
            video::IGPUDescriptorSet::SWriteDescriptorSet write;
            write.arrayElement = 0u;
            write.binding = 0u;
            write.count = 1u;
            write.descriptorType = asset::EDT_STORAGE_BUFFER;
            write.dstSet = gpu_ds2.get();
            video::IGPUDescriptorSet::SDescriptorInfo info[1];
            write.info = info;
            write.info->desc = gpuPrecomputedStuffSsbo;
            write.info->buffer.offset = 0u;
            write.info->buffer.size = UBO_SZ;
            driver->updateDescriptorSets(1u, &write, 0u, nullptr);
        }
    }

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.5f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.5f);
	camera->setFarValue(5000.0f);

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
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 2u, 1u, &gpu_ds2.get(), nullptr);
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