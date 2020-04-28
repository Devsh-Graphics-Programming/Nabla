#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include <irr/video/IGPUVirtualTexture.h>
#include <irr/asset/CMTLPipelineMetadata.h>
#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include <irr/asset/filters/CMipMapGenerationImageFilter.h>

//#include "../../ext/ScreenShot/ScreenShot.h"
using namespace irr;
using namespace core;

constexpr const char* GLSL_VT_DESCRIPTORS = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
#ifndef _NO_UV
#define VT_COUNT 3
#include <%s> //irr/builtin/glsl/virtual_texturing/descriptors.glsl/...
layout (set = 2, binding = 1, std430) readonly restrict buffer LUT
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
struct PCstruct
{
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
};
layout (push_constant) uniform Block {
    PCstruct params;
} PC;
#define _IRR_FRAG_PUSH_CONSTANTS_DEFINED_
)";
constexpr const char* GLSL_TEXTURE_SAMPLE_OVERRIDE =
R"(

mat2 dUV;//global

vec4 irr_sample_Ka(in vec2 uv) { return irr_glsl_textureVT(PC.params.map_Ka_data, uv, dUV); }

vec4 irr_sample_Kd(in vec2 uv) { return irr_glsl_textureVT(PC.params.map_Kd_data, uv, dUV); }

vec4 irr_sample_Ks(in vec2 uv) { return irr_glsl_textureVT(PC.params.map_Ks_data, uv, dUV); }

vec4 irr_sample_Ns(in vec2 uv) { return irr_glsl_textureVT(PC.params.map_Ns_data, uv, dUV); }

vec4 irr_sample_d(in vec2 uv) { return irr_glsl_textureVT(PC.params.map_d_data, uv, dUV); }

vec4 irr_sample_bump(in vec2 uv) { return irr_glsl_textureVT(PC.params.map_bump_data, uv, dUV); }

#define _IRR_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_

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

//irr/builtin/glsl/virtual_texturing/functions.glsl/...
#include <%s>
)";

constexpr const char* GLSL_FS_MAIN_OVERRIDE =
R"(
void main()
{
    dUV = mat2(dFdx(UV),dFdy(UV));//writing to global var
//#define COLOR_IS_DIFFUSE_TEX
#ifndef COLOR_IS_DIFFUSE_TEX
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
            d = irr_sample_d(UV).r;
            color *= d;
        }
#endif
        break;
    }
#else//!COLOR_IS_DIFFUSE_TEX
    vec3 color = vec3(0.0);
    float d = 1.0;
#ifndef _NO_UV
    if ((PC.params.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        color = irr_sample_Kd(UV).rgb;
    else
#endif
        color = PC.params.Kd;
#endif//!COLOR_IS_DIFFUSE_TEX
    OutColor = vec4(color, d);
}
#define _IRR_FRAG_MAIN_DEFINED_
)";

using STextureData = asset::ICPUVirtualTexture::STextureData;

constexpr uint32_t PGTAB_SZ_LOG2 = 7u;
constexpr uint32_t PGTAB_LAYERS_PER_FORMAT = 1u;
constexpr uint32_t PGTAB_LAYERS = 3u;
constexpr uint32_t PAGE_SZ_LOG2 = 7u;
constexpr uint32_t TILES_PER_DIM_LOG2 = 6u;
constexpr uint32_t PHYS_ADDR_TEX_LAYERS = 3u;
constexpr uint32_t PAGE_PADDING = 8u;
constexpr uint32_t MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u; //4096

constexpr uint32_t VT_SET = 0u;
constexpr uint32_t PGTAB_BINDING = 0u;
constexpr uint32_t PHYSICAL_STORAGE_VIEWS_BINDING = 1u;

STextureData getTextureData(const asset::ICPUImage* _img, asset::ICPUVirtualTexture* _vt, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
    const auto& extent = _img->getCreationParameters().extent;

    asset::IImage::SSubresourceRange subres;
    subres.baseMipLevel = 0u;
    subres.levelCount = core::findLSB(core::roundDownToPoT<uint32_t>(std::max(extent.width, extent.height))) + 1;

    return _vt->pack(_img, subres, _uwrap, _vwrap, _borderColor);
}

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

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedFragShader(const asset::ICPUSpecializedShader* _fs, const asset::ICPUVirtualTexture* _vt)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    std::string glsl = reinterpret_cast<const char*>( unspec->getSPVorGLSL()->getPointer() );
    size_t firstNewlineAfterVersion = glsl.find("\n",glsl.find("#version "));
    glsl.insert(firstNewlineAfterVersion, "\n#include <"+_vt->getGLSLExtensionsIncludePath()+">\n");
    glsl.insert(glsl.find("#ifndef _IRR_FRAG_PUSH_CONSTANTS_DEFINED_"), GLSL_PUSH_CONSTANTS_OVERRIDE);
    std::string str = GLSL_VT_DESCRIPTORS;
    str.resize(str.size()+500u, ' ');
    sprintf(str.data(), GLSL_VT_DESCRIPTORS, _vt->getGLSLDescriptorsIncludePath(VT_SET, PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING).c_str());
    glsl.insert(glsl.find("#if !defined(_IRR_FRAG_SET3_BINDINGS_DEFINED_)"), str.c_str());
    str = GLSL_VT_FUNCTIONS;
    str.resize(str.size()+500u, ' ');
    sprintf(str.data(), GLSL_VT_FUNCTIONS, _vt->getGLSLFunctionsIncludePath("getPgTabSzLog2", "getPhysPgTexSzRcp", "getVTexSzRcp", "getPhysicalIDForLayer").c_str());
    glsl.insert(glsl.find("#if !defined(_IRR_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_)"), str.c_str());
    glsl.insert(glsl.find("#if !defined(_IRR_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_)"), GLSL_TEXTURE_SAMPLE_OVERRIDE);
    glsl.insert(glsl.find("#ifndef _IRR_FRAG_MAIN_DEFINED_"), GLSL_FS_MAIN_OVERRIDE);

    auto* f = fopen("fs.glsl","w");
    fwrite(glsl.c_str(), 1, glsl.size(), f);
    fclose(f);

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str());
    auto specinfo = _fs->getSpecializationInfo();//intentional copy
    auto fsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return fsNew;
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

    constexpr uint32_t VT_COUNT = 3u;

    core::smart_refctd_ptr<asset::ICPUVirtualTexture> vt;
    {
        std::array<asset::ICPUVirtualTexture::ICPUVTResidentStorage::SCreationParams,VT_COUNT> storage;
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

        vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>(storage.data(), storage.size(), PGTAB_SZ_LOG2, PGTAB_LAYERS, PAGE_SZ_LOG2, PAGE_PADDING, MAX_ALLOCATABLE_TEX_SZ_LOG2);
    }

    core::unordered_map<core::smart_refctd_ptr<asset::ICPUImage>, STextureData> VTtexDataMap;
    core::unordered_map<core::smart_refctd_ptr<asset::ICPUSpecializedShader>, core::smart_refctd_ptr<asset::ICPUSpecializedShader>> modifiedShaders;

    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds0layout;
    {
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

        std::array<core::smart_refctd_ptr<asset::ICPUSampler>,VT_COUNT> samplers;
        std::fill(samplers.begin(), samplers.end(), sampler);

        params.AnisotropicFilter = 0u;
        params.MaxFilter = asset::ISampler::ETF_NEAREST;
        params.MinFilter = asset::ISampler::ETF_NEAREST;
        params.MipmapMode = asset::ISampler::ESMM_NEAREST;
        auto samplerPgt = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);

        auto bindings = vt->getDSlayoutBindings(PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);
        (*bindings)[0].samplers = &samplerPgt;
        (*bindings)[1].samplers = samplers.data();

        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings->data(), bindings->data()+bindings->size());
    }
    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds2layout;
    {
        std::array<asset::ICPUDescriptorSetLayout::SBinding,2> bnd;
        bnd[0].binding = 0u;
        bnd[0].count = 1u;
        bnd[0].samplers = nullptr;
        bnd[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bnd[0].type = asset::EDT_STORAGE_BUFFER;
        bnd[1].binding = 1u;
        bnd[1].count = 1u;
        bnd[1].samplers = nullptr;
        bnd[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bnd[1].type = asset::EDT_STORAGE_BUFFER;
        ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd.data(),bnd.data()+bnd.size());
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
        }
        auto* pipeline = mb->getPipeline();
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
                newfs = createModifiedFragShader(fs,vt.get());
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

    auto gpuvt = core::make_smart_refctd_ptr<video::IGPUVirtualTexture>(driver, am, vt.get());

    auto gpuds0layout = driver->getGPUObjectsFromAssets(&ds0layout.get(), &ds0layout.get()+1)->front();
    auto gpuds0 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuds0layout));//intentionally not moving layout
    {
        auto writes = gpuvt->getDescriptorSetWrites(gpuds0.get(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        driver->updateDescriptorSets(writes.first->size(), writes.first->data(), 0u, nullptr);
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
    auto gpuds2 = driver->createGPUDescriptorSet(std::move(gpu_ds2layout));
    core::smart_refctd_ptr<video::IGPUBuffer> ds2_ssbo;
    {
        constexpr uint32_t ENTRY_SZ = sizeof(uint32_t);
        constexpr uint32_t ENTRY_ALIGNMENT = sizeof(uint32_t);//change to 16 if std140
        constexpr uint32_t PRECOMPUTED_STUFF_COUNT = 3u;//pgtab_sz_log2, phys_pg_tex_sz_rcp, vtex_sz_rcp
        constexpr uint32_t PRECOMPUTED_STUFF_SZ = VT_COUNT*ENTRY_ALIGNMENT;
        constexpr uint32_t PRECOMPUTED_SSBO_SZ = PRECOMPUTED_STUFF_COUNT*PRECOMPUTED_STUFF_SZ;
        uint8_t precomputedStuff[PRECOMPUTED_SSBO_SZ]{};
        uint32_t* ptr = reinterpret_cast<uint32_t*>(precomputedStuff);
        uint32_t offset = 0u;
        //pgtab_sz_log2
        for (uint32_t i = 0u; i < VT_COUNT; ++i)
            ptr[i*ENTRY_ALIGNMENT/ENTRY_SZ] = core::findMSB(gpuvt->getPageTable()->getCreationParameters().extent.width);
        ptr += PRECOMPUTED_STUFF_SZ/ENTRY_SZ;
        //phys_pg_tex_sz_rcp
        for (uint32_t i = 0u; i < VT_COUNT; ++i)
        {
            const auto& storageImg = gpuvt->getFloatViews().begin()[i]->getCreationParameters().image;
            const double f = 1.0 / static_cast<double>(storageImg->getCreationParameters().extent.width);
            reinterpret_cast<float*>(ptr)[i*ENTRY_ALIGNMENT/ENTRY_SZ] = f;
        }
        ptr += PRECOMPUTED_STUFF_SZ/ENTRY_SZ;
        //vtex_sz_rcp
        for (uint32_t i = 0u; i < VT_COUNT; ++i)
        {
            double f = 1.0;
            f /= static_cast<double>(gpuvt->getPageTable()->getCreationParameters().extent.width);
            f /= static_cast<double>(gpuvt->getPageExtent());
            reinterpret_cast<float*>(ptr)[i*ENTRY_ALIGNMENT/ENTRY_SZ] = f;
        }

        const uint32_t lutSz = gpuvt->getLayerToViewIndexLUTBytesize();
        uint32_t lut[PGTAB_LAYERS]{};
        assert(sizeof(lut)>=lutSz);
        gpuvt->writeLayerToViewIndexLUTContents(lut);
        const uint32_t lutOffset = core::alignUp(PRECOMPUTED_SSBO_SZ, driver->getRequiredSSBOAlignment());
        ds2_ssbo = driver->createFilledDeviceLocalGPUBufferOnDedMem(lutOffset+lutSz, precomputedStuff);
        driver->updateBufferRangeViaStagingBuffer(ds2_ssbo.get(), lutOffset, lutSz, lut);

        {
            std::array<video::IGPUDescriptorSet::SWriteDescriptorSet,2> write;
            video::IGPUDescriptorSet::SDescriptorInfo info[2];

            write[0].arrayElement = 0u;
            write[0].binding = 0u;
            write[0].count = 1u;
            write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
            write[0].dstSet = gpuds2.get();
            write[0].info = info;
            write[0].info->desc = ds2_ssbo;
            write[0].info->buffer.offset = 0u;
            write[0].info->buffer.size = PRECOMPUTED_SSBO_SZ;

            write[1].arrayElement = 0u;
            write[1].binding = 1u;
            write[1].count = 1u;
            write[1].descriptorType = asset::EDT_STORAGE_BUFFER;
            write[1].dstSet = gpuds2.get();
            write[1].info = info+1;
            write[1].info->desc = ds2_ssbo;
            write[1].info->buffer.offset = lutOffset;
            write[1].info->buffer.size = lutSz;

            driver->updateDescriptorSets(write.size(), write.data(), 0u, nullptr);
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
            video::IGPUDescriptorSet* ds[] {gpuds0.get(),gpuds1.get(),gpuds2.get()};
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 3u, ds, nullptr);

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