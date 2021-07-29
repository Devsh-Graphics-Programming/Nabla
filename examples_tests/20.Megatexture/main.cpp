// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nbl/video/IGPUVirtualTexture.h>
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../source/Nabla/CFileSystem.h"

using namespace nbl;
using namespace core;

constexpr const char* SHADER_OVERRIDES = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
#ifndef _NO_UV
    #include <nbl/builtin/glsl/virtual_texturing/extensions.glsl>

    #define _NBL_VT_DESCRIPTOR_SET 0
    #define _NBL_VT_PAGE_TABLE_BINDING 0

    #define _NBL_VT_FLOAT_VIEWS_BINDING 1 
    #define _NBL_VT_FLOAT_VIEWS_COUNT %u
    #define _NBL_VT_FLOAT_VIEWS

    #define _NBL_VT_INT_VIEWS_BINDING 2
    #define _NBL_VT_INT_VIEWS_COUNT 0
    #define _NBL_VT_INT_VIEWS

    #define _NBL_VT_UINT_VIEWS_BINDING 3
    #define _NBL_VT_UINT_VIEWS_COUNT 0
    #define _NBL_VT_UINT_VIEWS
    #include <nbl/builtin/glsl/virtual_texturing/descriptors.glsl>

    layout (set = 2, binding = 0, std430) restrict readonly buffer PrecomputedStuffSSBO
    {
        uint pgtab_sz_log2;
        float vtex_sz_rcp;
        float phys_pg_tex_sz_rcp[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
        uint layer_to_sampler_ix[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
    } precomputed;
#endif
#define _NBL_FRAG_SET3_BINDINGS_DEFINED_

struct MaterialParams
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
    MaterialParams params;
} PC;
#define _NBL_FRAG_PUSH_CONSTANTS_DEFINED_

#include <nbl/builtin/glsl/loader/mtl/common.glsl>
nbl_glsl_MTLMaterialParameters nbl_glsl_getMaterialParameters() // this function is for MTL's shader only
{
    MaterialParams params = PC.params;

    nbl_glsl_MTLMaterialParameters mtl_params;
    mtl_params.Ka = params.Ka;
    mtl_params.Kd = params.Kd;
    mtl_params.Ks = params.Ks;
    mtl_params.Ke = params.Ke;
    mtl_params.Ns = params.Ns;
    mtl_params.d = params.d;
    mtl_params.Ni = params.Ni;
    mtl_params.extra = params.extra;
    return mtl_params;
}
#define _NBL_FRAG_GET_MATERIAL_PARAMETERS_FUNCTION_DEFINED_

#ifndef _NO_UV
    uint nbl_glsl_VT_layer2pid(in uint layer)
    {
        return precomputed.layer_to_sampler_ix[layer];
    }
    uint nbl_glsl_VT_getPgTabSzLog2()
    {
        return precomputed.pgtab_sz_log2;
    }
    float nbl_glsl_VT_getPhysPgTexSzRcp(in uint layer)
    {
        return precomputed.phys_pg_tex_sz_rcp[layer];
    }
    float nbl_glsl_VT_getVTexSzRcp()
    {
        return precomputed.vtex_sz_rcp;
    }
    #define _NBL_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_

    //nbl/builtin/glsl/virtual_texturing/functions.glsl/...
    #include <%s>
#endif


#ifndef _NO_UV
    vec4 nbl_sample_Ka(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ka_data, uv, dUV); }

    vec4 nbl_sample_Kd(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Kd_data, uv, dUV); }

    vec4 nbl_sample_Ks(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ks_data, uv, dUV); }

    vec4 nbl_sample_Ns(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ns_data, uv, dUV); }

    vec4 nbl_sample_d(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_d_data, uv, dUV); }

    vec4 nbl_sample_bump(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_bump_data, uv, dUV); }
#endif
#define _NBL_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_
)";

using STextureData = asset::ICPUVirtualTexture::SMasterTextureData;

constexpr uint32_t PAGE_SZ_LOG2 = 7u;
constexpr uint32_t TILES_PER_DIM_LOG2 = 4u;
constexpr uint32_t PAGE_PADDING = 8u;
constexpr uint32_t MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u; //4096

constexpr uint32_t VT_SET = 0u;
constexpr uint32_t PGTAB_BINDING = 0u;
constexpr uint32_t PHYSICAL_STORAGE_VIEWS_BINDING = 1u;

struct commit_t
{
    STextureData addr;
    core::smart_refctd_ptr<asset::ICPUImage> texture;
    asset::ICPUImage::SSubresourceRange subresource;
    asset::ICPUSampler::E_TEXTURE_CLAMP uwrap;
    asset::ICPUSampler::E_TEXTURE_CLAMP vwrap;
    asset::ICPUSampler::E_TEXTURE_BORDER_COLOR border;
};
STextureData getTextureData(core::vector<commit_t>& _out_commits, const asset::ICPUImage* _img, asset::ICPUVirtualTexture* _vt, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
    const auto& extent = _img->getCreationParameters().extent;

    auto imgAndOrigSz = asset::ICPUVirtualTexture::createPoTPaddedSquareImageWithMipLevels(_img, _uwrap, _vwrap, _borderColor);

    asset::IImage::SSubresourceRange subres;
    subres.baseMipLevel = 0u;
    subres.levelCount = core::findLSB(core::roundDownToPoT<uint32_t>(std::max(extent.width, extent.height))) + 1;
    subres.baseArrayLayer = 0u;
    subres.layerCount = 1u;

    auto addr = _vt->alloc(_img->getCreationParameters().format, imgAndOrigSz.second, subres, _uwrap, _vwrap);
    commit_t cm{ addr, std::move(imgAndOrigSz.first), subres, _uwrap, _vwrap, _borderColor };

    _out_commits.push_back(cm);

    return addr;
}

constexpr uint32_t TEX_OF_INTEREST_CNT = 6u;
#include "nbl/nblpack.h"
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
#include "nbl/nblunpack.h"
static_assert(sizeof(SPushConstants)<=asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "doesnt fit in push constants");

constexpr uint32_t texturesOfInterest[TEX_OF_INTEREST_CNT]{
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_AMBIENT,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_DIFFUSE,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_SPECULAR,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_SHININESS,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_OPACITY,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP
};

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedFragShader(const asset::ICPUSpecializedShader* _fs, const asset::ICPUVirtualTexture* _vt)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
    auto end = begin+unspec->getSPVorGLSL()->getSize();
    std::string glsl(begin,end);

    std::string prelude(strlen(SHADER_OVERRIDES)+500u,'\0');
    sprintf(prelude.data(), SHADER_OVERRIDES, _vt->getFloatViews().size(), _vt->getGLSLFunctionsIncludePath().c_str());
    prelude.resize(strlen(prelude.c_str()));

    size_t firstNewlineAfterVersion = glsl.find("\n",glsl.find("#version "));
    glsl.insert(firstNewlineAfterVersion, prelude);

    auto* f = fopen("fs.glsl","w");
    fwrite(glsl.c_str(), 1, glsl.size(), f);
    fclose(f);

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str());
    auto specinfo = _fs->getSpecializationInfo();//intentional copy
    auto fsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return fsNew;
}

/* TODO: add it after mergin Dan PR

class EventReceiver : public nbl::IEventReceiver
{
    _NBL_STATIC_INLINE_CONSTEXPR int32_t MAX_LOD = 8;
	public:
		bool OnEvent(const nbl::SEvent& event)
		{
			if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
			{
				switch (event.KeyInput.Key)
				{
					case nbl::KEY_KEY_Q: // switch wire frame mode
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

*/

int main()
{
    constexpr uint32_t WIN_W = 1280;
    constexpr uint32_t WIN_H = 720;
    constexpr uint32_t FBO_COUNT = 1u;

    auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "MeshLoaders");
    auto window = std::move(initOutput.window);
    auto gl = std::move(initOutput.apiConnection);
    auto surface = std::move(initOutput.surface);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queue = std::move(initOutput.queue);
    auto swapchain = std::move(initOutput.swapchain);
    auto renderpass = std::move(initOutput.renderpass);
    auto fbo = std::move(initOutput.fbo[0]);
    auto commandPool = std::move(initOutput.commandPool);

    core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffers[1];
    logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers);
    auto commandBuffer = commandBuffers[0];

    core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
    {
        nbl::core::smart_refctd_ptr<nbl::io::IFileSystem> fileSystem = nbl::core::make_smart_refctd_ptr<nbl::io::CFileSystem>("");
        assetManager = core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(std::move(fileSystem));
    }

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
    cpu2gpuParams.assetManager = assetManager.get();
    cpu2gpuParams.device = logicalDevice.get();
    cpu2gpuParams.finalQueueFamIx = queue->getFamilyIndex();
    cpu2gpuParams.limits = gpuPhysicalDevice->getLimits();
    cpu2gpuParams.pipelineCache = nullptr;
    cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;
    cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
    cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue;

    auto createDescriptorPool = [&](const uint32_t textureCount)
    {
        constexpr uint32_t maxItemCount = 256u;
        {
            nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
            poolSize.count = textureCount;
            poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
            return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
        }
    };

    core::smart_refctd_ptr<asset::ICPUVirtualTexture> vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>([](asset::E_FORMAT_CLASS) -> uint32_t { return TILES_PER_DIM_LOG2; }, PAGE_SZ_LOG2, PAGE_PADDING, MAX_ALLOCATABLE_TEX_SZ_LOG2);

    core::unordered_map<core::smart_refctd_ptr<asset::ICPUImage>, STextureData> VTtexDataMap;
    core::unordered_map<core::smart_refctd_ptr<asset::ICPUSpecializedShader>, core::smart_refctd_ptr<asset::ICPUSpecializedShader>> modifiedShaders;

    auto* fileSystem = assetManager->getFileSystem();
    fileSystem->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = assetManager->getAsset("sponza.obj", lp);
    assert(!meshes_bundle.getContents().empty());

    const auto meta = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    // all pipelines will have the same metadata
    const asset::CMTLMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;
    core::vector<commit_t> vt_commits;
    //modifying push constants and default fragment shader for VT
    for (auto mb : mesh_raw->getMeshBuffers())
    {
        SPushConstants pushConsts;
        memset(pushConsts.map_data, 0xff, TEX_OF_INTEREST_CNT*sizeof(pushConsts.map_data[0]));
        pushConsts.extra = 0u;

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
            STextureData texData = STextureData::invalid();
            auto found = VTtexDataMap.find(img);
            if (found != VTtexDataMap.end())
                texData = found->second;
            else {
                const asset::E_FORMAT fmt = img->getCreationParameters().format;
                texData = getTextureData(vt_commits, img.get(), vt.get(), uwrap, vwrap, borderColor);
                VTtexDataMap.insert({img,texData});
            }

            static_assert(sizeof(texData)==sizeof(pushConsts.map_data[0]), "wrong reinterpret_cast");
            pushConsts.map_data[k] = reinterpret_cast<uint64_t*>(&texData)[0];
        }

        pipelineMetadata = static_cast<const asset::CMTLMetadata::CRenderpassIndependentPipeline*>(meta->getAssetSpecificMetadata(mb->getPipeline()));

        //copy texture presence flags
        pushConsts.extra = pipelineMetadata->m_materialParams.extra;
        pushConsts.ambient = pipelineMetadata->m_materialParams.ambient;
        pushConsts.diffuse = pipelineMetadata->m_materialParams.diffuse;
        pushConsts.emissive = pipelineMetadata->m_materialParams.emissive;
        pushConsts.specular = pipelineMetadata->m_materialParams.specular;
        pushConsts.IoR = pipelineMetadata->m_materialParams.IoR;
        pushConsts.opacity = pipelineMetadata->m_materialParams.opacity;
        pushConsts.shininess = pipelineMetadata->m_materialParams.shininess;
        memcpy(mb->getPushConstantsDataPtr(), &pushConsts, sizeof(pushConsts));

        //we dont want this DS to be converted into GPU DS, so set to nullptr
        //dont worry about deletion of textures (invalidation of pointers), they're grabbed in VTtexDataMap
        mb->setAttachedDescriptorSet(nullptr);
    }
    assert(pipelineMetadata);

    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds0layout;
    {
        auto sizes = vt->getDSlayoutBindings(nullptr, nullptr);
        auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSetLayout::SBinding>>(sizes.first);
        auto samplers = core::make_refctd_dynamic_array< core::smart_refctd_dynamic_array<core::smart_refctd_ptr<asset::ICPUSampler>>>(sizes.second);

        vt->getDSlayoutBindings(bindings->data(), samplers->data(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings->data(), bindings->data() + bindings->size());
    }
    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds2layout;
    {
        std::array<asset::ICPUDescriptorSetLayout::SBinding, 1> bnd;
        bnd[0].binding = 0u;
        bnd[0].count = 1u;
        bnd[0].samplers = nullptr;
        bnd[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        bnd[0].type = asset::EDT_STORAGE_BUFFER;
        ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd.data(), bnd.data() + bnd.size());
    }

    core::smart_refctd_ptr<asset::ICPUPipelineLayout> pipelineLayout;
    {
        asset::SPushConstantRange pcrng;
        pcrng.offset = 0;
        pcrng.size = 128;
        pcrng.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;

        pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(&pcrng, &pcrng + 1, core::smart_refctd_ptr(ds0layout), nullptr, core::smart_refctd_ptr(ds2layout), nullptr);
    }

    for (auto mb : mesh_raw->getMeshBuffers())
    {
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
                newfs = createModifiedFragShader(fs, vt.get());
                modifiedShaders.insert({ core::smart_refctd_ptr<asset::ICPUSpecializedShader>(fs),newfs });
            }
            newPipeline->setShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, newfs.get());
        }

        //set new pipeline (with overriden FS and layout)
        mb->setPipeline(std::move(newPipeline));
    }

    vt->shrink();
    for (const auto& cm : vt_commits)
    {
        vt->commit(cm.addr, cm.texture.get(), cm.subresource, cm.uwrap, cm.vwrap, cm.border);
    }

    auto gpuvt = core::make_smart_refctd_ptr<video::IGPUVirtualTexture>(logicalDevice.get(), queue, vt.get());

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds0layout;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds0layout.get(), &ds0layout.get() + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        gpuds0layout = (*gpu_array)[0];
    }

    auto descriptorPoolDs0 = createDescriptorPool(1u); // TODO check it out

    auto gpuds0 = logicalDevice->createGPUDescriptorSet(descriptorPoolDs0.get(), core::smart_refctd_ptr(gpuds0layout));//intentionally not moving layout
    {
        auto sizes = gpuvt->getDescriptorSetWrites(nullptr, nullptr, nullptr);
        auto writes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<video::IGPUDescriptorSet::SWriteDescriptorSet>>(sizes.first);
        auto info = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<video::IGPUDescriptorSet::SDescriptorInfo>>(sizes.second);

        gpuvt->getDescriptorSetWrites(writes->data(), info->data(), gpuds0.get(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        logicalDevice->updateDescriptorSets(writes->size(), writes->data(), 0u, nullptr);
    }

    //we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
    //so we can create just one DS
    
    asset::ICPUDescriptorSetLayout* ds1layout = mesh_raw->getMeshBuffers().begin()[0]->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
        {
            ds1UboBinding = bnd.binding;
            break;
        }

    size_t neededDS1UBOsz = 0ull;
    {
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
    }

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        gpuds1layout = (*gpu_array)[0];
    }

    auto ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
    ubomemreq.vulkanReqs.size = neededDS1UBOsz;
    auto gpuubo = logicalDevice->createGPUBufferOnDedMem(ubomemreq, true);

    auto descriptorPoolDs1 = createDescriptorPool(1u); // TODO check it out

    auto gpuds1 = logicalDevice->createGPUDescriptorSet(descriptorPoolDs1.get(), std::move(gpuds1layout));
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
        logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
    }

    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&mesh_raw, &mesh_raw + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        gpumesh = (*gpu_array)[0];
    }

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpu_ds2layout;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds2layout.get(), &ds2layout.get() + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        gpu_ds2layout = (*gpu_array)[0];
    }

    auto descriptorPoolDs2 = createDescriptorPool(1u); // TODO check it out
    
    auto gpuds2 = logicalDevice->createGPUDescriptorSet(descriptorPoolDs2.get(), std::move(gpu_ds2layout));
    {
        core::smart_refctd_ptr<video::IGPUBuffer> buffer = logicalDevice->createFilledDeviceLocalGPUBufferOnDedMem(queue, sizeof(video::IGPUVirtualTexture::SPrecomputedData), &gpuvt->getPrecomputedData());
        {
            std::array<video::IGPUDescriptorSet::SWriteDescriptorSet,1> write;
            video::IGPUDescriptorSet::SDescriptorInfo info[1];

            write[0].arrayElement = 0u;
            write[0].binding = 0u;
            write[0].count = 1u;
            write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
            write[0].dstSet = gpuds2.get();
            write[0].info = info;
            write[0].info->desc = buffer;
            write[0].info->buffer.offset = 0u;
            write[0].info->buffer.size = sizeof(video::IGPUVirtualTexture::SPrecomputedData);

            logicalDevice->updateDescriptorSets(write.size(), write.data(), 0u, nullptr);
        }
    }

    using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
    std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
    {
        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
        {
            auto gpuIndependentPipeline = gpumesh->getMeshBuffers().begin()[i]->getPipeline();

            nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
            graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
            graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

            const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
            gpuPipelines[adress] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
        }
    }

    core::vectorSIMDf cameraPosition(-1, 2, -10);
    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
    matrix3x4SIMD viewMatrix = matrix3x4SIMD::buildCameraLookAtMatrixLH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
    auto viewProjectionMatrix = matrix4SIMD::concatenateBFollowedByA(projectionMatrix, matrix4SIMD(viewMatrix));

	while(true)
	{
        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(0);

        asset::SViewport viewport;
        viewport.minDepth = 1.f;
        viewport.maxDepth = 0.f;
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
        commandBuffer->setViewport(0u, 1u, &viewport);

        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        nbl::asset::VkRect2D area;
        area.offset = { 0,0 };
        area.extent = { WIN_W, WIN_H };
        nbl::asset::SClearValue clear;
        clear.color.float32[0] = 1.f;
        clear.color.float32[1] = 1.f;
        clear.color.float32[2] = 1.f;
        clear.color.float32[3] = 1.f;
        beginInfo.clearValueCount = 1u;
        beginInfo.framebuffer = fbo;
        beginInfo.renderpass = renderpass;
        beginInfo.renderArea = area;
        beginInfo.clearValues = &clear;

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));

        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        core::vector<uint8_t> uboData(gpuubo->getSize());
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                    {
                        memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                    } break;

                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
                    {
                        memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                    } break;

                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                    {
                        memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                    } break;
                }
            }
        }

        commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
        {
            auto gpuMeshBuffer = gpumesh->getMeshBuffers().begin()[i];
            auto gpuGraphicsPipeline = gpuPipelines[reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];
            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();

            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

            video::IGPUDescriptorSet* gpuDescriptorSets[] {gpuds0.get(),gpuds1.get(),gpuds2.get()};
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 0u, 3u, gpuDescriptorSets, nullptr);
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        commandBuffer->endRenderPass();
        commandBuffer->end();

        auto img_acq_sem = logicalDevice->createSemaphore();
        auto render_finished_sem = logicalDevice->createSemaphore();

        uint32_t imgnum = 0u;
        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; // ns
        swapchain->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queue, img_acq_sem.get(), render_finished_sem.get());
        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queue, render_finished_sem.get(), imgnum);
	}

	return 0;
}