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

constexpr const char* GLSL_VT_FUNCTIONS =
R"(
vec4 textureVT(in uvec2 _texData, in vec2 uv, in mat2 dUV)
{
    vec2 scale = unpackSize(_texData);
    vec2 virtualUV = unpackVirtualUV(_texData);
    virtualUV += scale*uv;
    return vTextureGrad(virtualUV, dUV, scale*float(PAGE_SZ)*vec2(textureSize(pgTabTex[1],0)));
}
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

    core::smart_refctd_ptr<video::IGPUTexturePacker> gpuTexPackers[ETP_COUNT];
    for (uint32_t i = 0u; i < ETP_COUNT; ++i)
        gpuTexPackers[i] = core::make_smart_refctd_ptr<video::IGPUTexturePacker>(driver, texPackers[i].get());

    //TODO ds0 will most likely also get some UBO with data for MDI (instead of using push constants)
    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds0layout;
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
        auto sampler = driver->createGPUSampler(params);

        std::array<core::smart_refctd_ptr<video::IGPUSampler>,ETP_COUNT> samplers;
        std::fill(samplers.begin(), samplers.end(), sampler);

        params.AnisotropicFilter = 0u;
        params.MaxFilter = asset::ISampler::ETF_NEAREST;
        params.MinFilter = asset::ISampler::ETF_NEAREST;
        params.MipmapMode = asset::ISampler::ESMM_NEAREST;
        auto samplerPgt = driver->createGPUSampler(params);

        std::array<core::smart_refctd_ptr<video::IGPUSampler>, ETP_COUNT> samplersPgt;
        std::fill(samplersPgt.begin(), samplersPgt.end(), samplerPgt);

        std::array<video::IGPUDescriptorSetLayout::SBinding, 2> bindings;
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

        //ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings.data(), bindings.data()+bindings.size());
        gpuds0layout = core::make_smart_refctd_ptr<video::IGPUDescriptorSetLayout>(bindings.data(), bindings.data()+bindings.size());
    }
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
        // TODO: loop over all allocated textures
        {
            // TODO: set viewport to match texture placing in Virtual Address Space
            // TODO: feed the STextureData for the allocation of that texture using push constants
            driver->drawMeshBuffer(fsTriangleMeshBuffer.get());
        }

		driver->endScene();
	}

	return 0;
}