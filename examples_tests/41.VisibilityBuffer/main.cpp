// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

#include "common.h"

//vt stuff
using STextureData = asset::ICPUVirtualTexture::SMasterTextureData;

constexpr uint32_t TILES_PER_DIM_LOG2 = 4u;
constexpr uint32_t MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u; //4096

struct commit_t
{
    STextureData addr;
    core::smart_refctd_ptr<asset::ICPUImage> texture;
    asset::ICPUImage::SSubresourceRange subresource;
    asset::ICPUSampler::E_TEXTURE_CLAMP uwrap;
    asset::ICPUSampler::E_TEXTURE_CLAMP vwrap;
    asset::ICPUSampler::E_TEXTURE_BORDER_COLOR border;
};

constexpr uint32_t TEX_OF_INTEREST_CNT = 6u;
constexpr uint32_t texturesOfInterest[TEX_OF_INTEREST_CNT] =
{
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_AMBIENT,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_DIFFUSE,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_SPECULAR,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_SHININESS,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_OPACITY,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP
};

#include "nbl/nblpack.h"
struct BatchInstanceData
{
    BatchInstanceData() : shininess(32.f), opacity(1.f), IoR(1.6f), extra(0u)
    {
        ambient = vector3df_SIMD(0.f);
        diffuse = vector3df_SIMD(1.f);
        specular = vector3df_SIMD(0.f);
        emissive = vector3df_SIMD(0.f);
        const auto invalid_tex = STextureData::invalid();
        std::fill_n(map_data,TEX_OF_INTEREST_CNT,reinterpret_cast<const uint64_t&>(invalid_tex));
    }
    BatchInstanceData(const BatchInstanceData& other) : BatchInstanceData()
    {
        this->operator=(other);
    }
    ~BatchInstanceData()
    {
    }

    BatchInstanceData& operator=(const BatchInstanceData& other)
    {
        ambient = other.ambient;
        diffuse = other.diffuse;
        specular = other.specular;
        emissive = other.emissive;
        std::copy_n(other.map_data,TEX_OF_INTEREST_CNT,map_data);
        shininess = other.shininess;
        opacity = other.opacity;
        IoR = other.IoR;
        extra = other.extra;
        return *this;
    }

    union
    {
        //Ka
        vector3df_SIMD ambient;
        struct
        {
            uint32_t invalid_0[3];
            uint32_t firstIndex;
        };
    };
    union
    {
        //Kd
        vector3df_SIMD diffuse;
        struct
        {
            uint32_t invalid_1[3];
            uint32_t vAttrPos;
        };
    };
    union
    {
        //Ks
        vector3df_SIMD specular;
        struct
        {
            uint32_t invalid_2[3];
            uint32_t vAttrUV;
        };
    };
    union
    {
        //Ke
        vector3df_SIMD emissive;
        struct
        {
            uint32_t invalid_3[3];
            uint32_t vAttrNormal;
        };
    };
    uint64_t map_data[TEX_OF_INTEREST_CNT];
    //Ns, specular exponent in phong model
    float shininess;
    //d
    float opacity;
    //Ni, index of refraction
    float IoR;
    uint32_t extra;
} PACK_STRUCT;
#include "nbl/nblunpack.h"
static_assert(sizeof(BatchInstanceData) <= asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "doesnt fit in push constants");

//mesh packing stuff
struct DrawIndexedIndirectInput
{
    size_t offset = 0u;
    size_t maxCount = 0u;

    static constexpr asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
    static constexpr asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
};


struct SceneData
{
    smart_refctd_ptr<IGPURenderpassIndependentPipeline> fillVBufferPpln;
    smart_refctd_ptr<IGPUComputePipeline> shadeVBufferPpln;

    smart_refctd_ptr<IGPUBuffer> mdiBuffer,idxBuffer;
    smart_refctd_ptr<IGPUDescriptorSet> vtDS,vgDS,perFrameDS,shadingDS;

    core::vector<DrawIndexedIndirectInput> drawIndirectInput;
    core::vector<uint32_t> pushConstantsData;

    smart_refctd_ptr<IGPUBuffer> ubo;
};

using MeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;
using GPUMeshPacker = CGPUMeshPackerV2<DrawElementsIndirectCommand_t>;

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

constexpr bool useSSBO = true;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
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
    auto* fs = am->getFileSystem();

    //
    auto createScreenSizedImage = [driver,&params](const E_FORMAT format) -> auto
    {
        IGPUImage::SCreationParams param;
        param.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
        param.type = IImage::ET_2D;
        param.format = format;
        param.extent = {params.WindowSize.Width,params.WindowSize.Height,1u};
        param.mipLevels = 1u;
        param.arrayLayers = 1u;
        param.samples = IImage::ESCF_1_BIT;
        return driver->createDeviceLocalGPUImageOnDedMem(std::move(param));
    };
    auto framebuffer = createScreenSizedImage(EF_R8G8B8A8_SRGB);
    auto createImageView = [driver,&params](smart_refctd_ptr<IGPUImage>&& image, const E_FORMAT format=EF_UNKNOWN) -> auto
    {
        IGPUImageView::SCreationParams params;
        params.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
        params.image = std::move(image);
        params.viewType = IGPUImageView::ET_2D;
        params.format = format!=EF_UNKNOWN ? format:params.image->getCreationParameters().format;
        params.components = {};
        params.subresourceRange = {};
        params.subresourceRange.levelCount = 1u;
        params.subresourceRange.layerCount = 1u;
        return driver->createGPUImageView(std::move(params));
    };
    auto depthBufferView = createImageView(createScreenSizedImage(EF_D32_SFLOAT));
    auto visBufferView = createImageView(createScreenSizedImage(EF_R32G32B32A32_UINT));

    auto visBuffer = driver->addFrameBuffer();
    visBuffer->attach(EFAP_DEPTH_ATTACHMENT,smart_refctd_ptr(depthBufferView));
    visBuffer->attach(EFAP_COLOR_ATTACHMENT0,smart_refctd_ptr(visBufferView));
    auto fb = driver->addFrameBuffer();
    fb->attach(EFAP_COLOR_ATTACHMENT0,createImageView(smart_refctd_ptr(framebuffer)));

    //
    SceneData sceneData;
    {
        //
        smart_refctd_ptr<IGPUDescriptorSetLayout> perFrameDSLayout,shadingDSLayout;
        {
            {
                IGPUDescriptorSetLayout::SBinding bindings[1];
                bindings[0].binding = 0u;
                bindings[0].count = 1u;
                bindings[0].samplers = nullptr;
                bindings[0].stageFlags = ISpecializedShader::ESS_VERTEX;
                bindings[0].type = EDT_UNIFORM_BUFFER;

                perFrameDSLayout = driver->createGPUDescriptorSetLayout(bindings,bindings+sizeof(bindings)/sizeof(IGPUDescriptorSetLayout::SBinding));
            }
            {
                sceneData.ubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));
                IGPUDescriptorSet::SDescriptorInfo infos[1];
                infos[0].desc = core::smart_refctd_ptr(sceneData.ubo);
                infos[0].buffer.offset = 0u;
                infos[0].buffer.size = sceneData.ubo->getSize();

                sceneData.perFrameDS = driver->createGPUDescriptorSet(smart_refctd_ptr(perFrameDSLayout));
                IGPUDescriptorSet::SWriteDescriptorSet writes[1];
                writes[0].dstSet = sceneData.perFrameDS.get();
                writes[0].binding = 0u;
                writes[0].arrayElement = 0u;
                writes[0].count = 1u;
                writes[0].descriptorType = EDT_UNIFORM_BUFFER;
                writes[0].info = infos+0u;
                driver->updateDescriptorSets(sizeof(writes)/sizeof(IGPUDescriptorSet::SWriteDescriptorSet),writes,0u,nullptr);
            }
            
            {
                IGPUSampler::SParams params;
                params.TextureWrapU = ISampler::ETC_MIRROR;
                params.TextureWrapV = ISampler::ETC_MIRROR;
                params.TextureWrapW = ISampler::ETC_MIRROR;
                params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
                params.MinFilter = ISampler::ETF_NEAREST;
                params.MaxFilter = ISampler::ETF_NEAREST;
                params.MipmapMode = ISampler::ESMM_NEAREST;
                params.AnisotropicFilter = 0;
                params.CompareEnable = 0;
                auto sampler = driver->createGPUSampler(params);

                IGPUDescriptorSetLayout::SBinding bindings[2];
                bindings[0].binding = 0u;
                bindings[0].count = 1u;
                bindings[0].samplers = &sampler;
                bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
                bindings[0].type = EDT_COMBINED_IMAGE_SAMPLER;
                bindings[1].binding = 1u;
                bindings[1].count = 1u;
                bindings[1].samplers = nullptr;
                bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
                bindings[1].type = EDT_STORAGE_IMAGE;

                shadingDSLayout = driver->createGPUDescriptorSetLayout(bindings,bindings+sizeof(bindings)/sizeof(IGPUDescriptorSetLayout::SBinding));
            }
            {
                IGPUDescriptorSet::SDescriptorInfo infos[2];
                infos[0].desc = core::smart_refctd_ptr(visBufferView);
                //infos[0].image.imageLayout = ?;
                infos[0].image.sampler = nullptr; // used immutable in the layout
                infos[1].desc = createImageView(std::move(framebuffer),EF_R8G8B8A8_UNORM);
                //infos[0].image.imageLayout = ?;
                infos[1].image.sampler = nullptr; // storage image

                sceneData.shadingDS = driver->createGPUDescriptorSet(smart_refctd_ptr(shadingDSLayout));
                IGPUDescriptorSet::SWriteDescriptorSet writes[2];
                for (auto i=0u; i<2u; i++)
                {
                    writes[i].dstSet = sceneData.shadingDS.get();
                    writes[i].binding = i;
                    writes[i].arrayElement = 0u;
                    writes[i].count = 1u;
                    writes[i].info = infos+i;
                }
                writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
                writes[1].descriptorType = EDT_STORAGE_IMAGE;
                driver->updateDescriptorSets(sizeof(writes)/sizeof(IGPUDescriptorSet::SWriteDescriptorSet),writes,0u,nullptr);
            }
        }

        //
        auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
        //loading cache from file
        qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse", true);

        // register the zip
        device->getFileSystem()->addFileArchive("../../media/sponza.zip");
        asset::IAssetLoader::SAssetLoadParams lp;
        auto meshes_bundle = am->getAsset("sponza.obj", lp);
        assert(!meshes_bundle.getContents().empty());
        auto mesh_raw = static_cast<asset::ICPUMesh*>(meshes_bundle.getContents().begin()->get());

        // ensure memory will be freed as soon as CPU assets are dropped
        am->clearAllAssetCache();
        //saving cache to file
        qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
        //qnc->clearCache<asset::EF_A2B10G10R10_SNORM_PACK32>(); // TODO

        //
        auto meshBuffers = mesh_raw->getMeshBufferVector();

        auto pipelineMeshBufferRanges = [&meshBuffers]() -> auto
        {
            core::vector<const core::smart_refctd_ptr<ICPUMeshBuffer>*> output;
            if (!meshBuffers.empty())
            {
                // sort meshbuffers by pipeline
                std::sort(meshBuffers.begin(),meshBuffers.end(),[](const auto& lhs, const auto& rhs)
                    {
                        auto lPpln = lhs->getPipeline();
                        auto rPpln = rhs->getPipeline();
                        // render non-transparent things first
                        if (lPpln->getBlendParams().blendParams[0].blendEnable < rPpln->getBlendParams().blendParams[0].blendEnable)
                            return true;
                        if (lPpln->getBlendParams().blendParams[0].blendEnable == rPpln->getBlendParams().blendParams[0].blendEnable)
                            return lPpln < rPpln;
                        return false;
                    }
                );

                const ICPURenderpassIndependentPipeline* mbPipeline = nullptr;
                for (const auto& mb : meshBuffers)
                if (mb->getPipeline()!=mbPipeline)
                {
                    mbPipeline = mb->getPipeline();
                    output.push_back(&mb);
                }
                output.push_back(meshBuffers.data()+meshBuffers.size());
            }
            return output;
        }();
        
        // the texture packing
        smart_refctd_ptr<IGPUVirtualTexture> gpuvt;
        {
            smart_refctd_ptr<ICPUVirtualTexture> vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>([](asset::E_FORMAT_CLASS) -> uint32_t { return TILES_PER_DIM_LOG2; }, PAGE_SZ_LOG2, PAGE_PADDING, MAX_ALLOCATABLE_TEX_SZ_LOG2);
            {
                core::vector<commit_t> vt_commits;

                core::unordered_map<smart_refctd_ptr<const asset::ICPUImage>,STextureData> VTtexDataMap;
                //modifying push constants and default fragment shader for VT
                for (const auto& meshbuffer : meshBuffers)
                {
                    BatchInstanceData pushConsts;

                    auto* ds = meshbuffer->getAttachedDescriptorSet();
                    if (!ds)
                        continue;

                    std::for_each_n(texturesOfInterest,TEX_OF_INTEREST_CNT,[&](auto textureOfInterest)
                    {
                        const auto* view = static_cast<asset::ICPUImageView*>(ds->getDescriptors(textureOfInterest).begin()->desc.get());
                        auto img = view->getCreationParameters().image;
                        auto extent = img->getCreationParameters().extent;
                        if (extent.width<=2u||extent.height<=2u) //dummy 2x2, TODO: compare by finding it in the cache!
                            return;

                        auto& texData = reinterpret_cast<STextureData&>(pushConsts.map_data[textureOfInterest]);
                        static_assert(sizeof(STextureData)==sizeof(pushConsts.map_data[0]),"wrong reinterpret_cast");

                        auto found = VTtexDataMap.find(img);
                        if (found!=VTtexDataMap.end())
                            texData = found->second;
                        else
                        {
                            const auto* smplr = ds->getLayout()->getBindings().begin()[textureOfInterest].samplers[0].get();
                            const auto uwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(smplr->getParams().TextureWrapU);
                            const auto vwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(smplr->getParams().TextureWrapV);
                            const auto borderColor = static_cast<asset::ISampler::E_TEXTURE_BORDER_COLOR>(smplr->getParams().BorderColor);
                            texData = getTextureData(vt_commits,img.get(),vt.get(),uwrap,vwrap,borderColor);
                            VTtexDataMap.insert({img,texData});
                            // get rid of pixel storage
                            img->convertToDummyObject(~0ull);
                        }
                    });

                    // all pipelines will have the same metadata
                    const asset::CMTLMetadata::CRenderpassIndependentPipeline* pipelineMetadata = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>()->getAssetSpecificMetadata(meshbuffer->getPipeline());
                    assert(pipelineMetadata);

                    //copy texture presence flags
                    pushConsts.extra = pipelineMetadata->m_materialParams.extra;
                    pushConsts.ambient = pipelineMetadata->m_materialParams.ambient;
                    pushConsts.diffuse = pipelineMetadata->m_materialParams.diffuse;
                    pushConsts.emissive = pipelineMetadata->m_materialParams.emissive;
                    pushConsts.specular = pipelineMetadata->m_materialParams.specular;
                    pushConsts.IoR = pipelineMetadata->m_materialParams.IoR;
                    pushConsts.opacity = pipelineMetadata->m_materialParams.opacity;
                    pushConsts.shininess = pipelineMetadata->m_materialParams.shininess;
                    memcpy(meshbuffer->getPushConstantsDataPtr(),&pushConsts,sizeof(pushConsts));

                    //we dont want this DS to be converted into GPU DS, so set to nullptr
                    //dont worry about deletion of textures (invalidation of pointers), they're grabbed in VTtexDataMap
                    meshbuffer->setAttachedDescriptorSet(nullptr);
                }

                vt->shrink();
                for (const auto& cm : vt_commits)
                {
                    vt->commit(cm.addr, cm.texture.get(), cm.subresource, cm.uwrap, cm.vwrap, cm.border);
                }
            }

            gpuvt = core::make_smart_refctd_ptr<IGPUVirtualTexture>(driver, vt.get());
        }

        // the vertex packing
        smart_refctd_ptr<GPUMeshPacker> gpump;
        smart_refctd_ptr<IGPUBuffer> batchDataSSBO;
        {
            constexpr uint16_t minTrisBatch = 256u; 
            constexpr uint16_t maxTrisBatch = MAX_TRIANGLES_IN_BATCH;

            constexpr uint32_t kVerticesPerTriangle = 3u;
            MeshPacker::AllocationParams allocParams;
            allocParams.indexBuffSupportedCnt = 32u*1024u*1024u;
            allocParams.indexBufferMinAllocCnt = minTrisBatch*kVerticesPerTriangle;
            allocParams.vertexBuffSupportedByteSize = 128u*1024u*1024u;
            allocParams.vertexBufferMinAllocByteSize = minTrisBatch;
            allocParams.MDIDataBuffSupportedCnt = 8192u;
            allocParams.MDIDataBuffMinAllocCnt = 16u;
            
            auto wholeMbRangeBegin = pipelineMeshBufferRanges.front();
            auto wholeMbRangeEnd = pipelineMeshBufferRanges.back();

            IMeshPackerV2Base::SupportedFormatsContainer formats;
            formats.insertFormatsFromMeshBufferRange(wholeMbRangeBegin, wholeMbRangeEnd);

            auto mp = core::make_smart_refctd_ptr<CCPUMeshPackerV2<>>(allocParams,formats,minTrisBatch,maxTrisBatch);
            
            const uint32_t mdiCntBound = mp->calcMDIStructMaxCount(wholeMbRangeBegin,wholeMbRangeEnd);

            auto allocData = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<MeshPacker::ReservedAllocationMeshBuffers>>(mdiCntBound);
            auto allocDataIt = allocData->begin();
            for (auto it=pipelineMeshBufferRanges.begin(); it!=pipelineMeshBufferRanges.end()-1u; )
            {
                auto mbRangeBegin = &(*it)->get();
                auto mbRangeEnd = &(*(++it))->get();

                bool allocSuccessfull = mp->alloc(&*allocDataIt,mbRangeBegin,mbRangeEnd);
                if (!allocSuccessfull)
                {
                    std::cout << "Alloc failed \n";
                    _NBL_DEBUG_BREAK_IF(true);
                }
                allocDataIt += mp->calcMDIStructMaxCount(mbRangeBegin,mbRangeEnd);
            }
            mp->shrinkOutputBuffersSize();
            mp->instantiateDataStorage();

            core::vector<BatchInstanceData> batchData;
            batchData.reserve(mdiCntBound);

            allocDataIt = allocData->begin();
            uint32_t mdiListOffset = 0u;
            for (auto it=pipelineMeshBufferRanges.begin(); it!=pipelineMeshBufferRanges.end()-1u; )
            {
                auto mbRangeBegin = &(*it)->get();
                auto mbRangeEnd = &(*(++it))->get();

                const uint32_t meshMdiBound = mp->calcMDIStructMaxCount(mbRangeBegin,mbRangeEnd);
                core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(std::distance(mbRangeBegin,mbRangeEnd));
                core::vector<MeshPacker::CombinedDataOffsetTable> cdot(meshMdiBound);
                uint32_t actualMdiCnt = mp->commit(pmbd.data(),cdot.data(),&*allocDataIt,mbRangeBegin,mbRangeEnd);
                allocDataIt += meshMdiBound;

                if (actualMdiCnt==0u)
                {
                    std::cout << "Commit failed \n";
                    _NBL_DEBUG_BREAK_IF(true);
                }

                sceneData.pushConstantsData.push_back(mdiListOffset);
                mdiListOffset += actualMdiCnt;

                DrawIndexedIndirectInput& mdiCallInput = sceneData.drawIndirectInput.emplace_back();
                mdiCallInput.maxCount = actualMdiCnt;
                mdiCallInput.offset = pmbd.front().mdiParameterOffset*sizeof(DrawElementsIndirectCommand_t);

                auto pmbdIt = pmbd.begin();
                auto cdotIt = cdot.begin();
                for (auto mbIt=mbRangeBegin; mbIt!=mbRangeEnd; mbIt++)
                {
                    const auto& material = *reinterpret_cast<BatchInstanceData*>((*mbIt)->getPushConstantsDataPtr());
                    const IMeshPackerBase::PackedMeshBufferData& packedData = *(pmbdIt++);
                    for (uint32_t mdi=0u; mdi<packedData.mdiParameterCount; mdi++)
                    {
                        auto& batch = batchData.emplace_back();
                        batch = material;
                        batch.firstIndex = reinterpret_cast<const DrawElementsIndirectCommand_t*>(mp->getPackerDataStore().MDIDataBuffer->getPointer())[packedData.mdiParameterOffset+mdi].firstIndex;

                        MeshPacker::CombinedDataOffsetTable& virtualAttribTable = *(cdotIt++);
                        constexpr auto UVAttributeIx = 2;
                        batch.vAttrPos = reinterpret_cast<const uint32_t&>(virtualAttribTable.attribInfo[(*mbIt)->getPositionAttributeIx()]);
                        batch.vAttrUV = reinterpret_cast<const uint32_t&>(virtualAttribTable.attribInfo[UVAttributeIx]);
                        batch.vAttrNormal = reinterpret_cast<const uint32_t&>(virtualAttribTable.attribInfo[(*mbIt)->getNormalAttributeIx()]);
                    }
                }
            }
            batchDataSSBO = driver->createFilledDeviceLocalGPUBufferOnDedMem(batchData.size()*sizeof(BatchInstanceData),batchData.data());

            gpump = core::make_smart_refctd_ptr<CGPUMeshPackerV2<>>(driver,mp.get());
            sceneData.mdiBuffer = gpump->getPackerDataStore().MDIDataBuffer;
            sceneData.idxBuffer = gpump->getPackerDataStore().indexBuffer;
        }
        mesh_raw->convertToDummyObject(~0u);

        //
        smart_refctd_ptr<IGPUDescriptorSetLayout> vtDSLayout;
        {
            // layout
            auto binding_and_sampler_count = gpuvt->getDSlayoutBindings(nullptr,nullptr);
            core::vector<IGPUDescriptorSetLayout::SBinding> vtBindings(binding_and_sampler_count.first);
            core::vector<smart_refctd_ptr<IGPUSampler>> vtSamplers(binding_and_sampler_count.second);
            gpuvt->getDSlayoutBindings(vtBindings.data(),vtSamplers.data(),_NBL_VT_PAGE_TABLE_BINDING,_NBL_VT_FLOAT_VIEWS_BINDING);
            auto& precomputedBinding = vtBindings.emplace_back();
            precomputedBinding.binding = 2u;
            precomputedBinding.type = EDT_STORAGE_BUFFER;
            precomputedBinding.count = 1u;
            precomputedBinding.samplers = nullptr; // ssbo
            for (auto& binding : vtBindings)
                binding.stageFlags = static_cast<ISpecializedShader::E_SHADER_STAGE>(ISpecializedShader::ESS_COMPUTE|ISpecializedShader::ESS_FRAGMENT);

            vtDSLayout = driver->createGPUDescriptorSetLayout(vtBindings.data(),vtBindings.data()+vtBindings.size());

            // write
            sceneData.vtDS = driver->createGPUDescriptorSet(smart_refctd_ptr(vtDSLayout));

            const auto sizesVT = gpuvt->getDescriptorSetWrites(nullptr,nullptr,nullptr);
            core::vector<video::IGPUDescriptorSet::SWriteDescriptorSet> writesVT(sizesVT.first+1u);
            core::vector<video::IGPUDescriptorSet::SDescriptorInfo> infoVT(sizesVT.second+1u);
            gpuvt->getDescriptorSetWrites(writesVT.data(),infoVT.data(),sceneData.vtDS.get(),_NBL_VT_PAGE_TABLE_BINDING,_NBL_VT_FLOAT_VIEWS_BINDING);

            auto& precomp = gpuvt->getPrecomputedData();
            infoVT.back().desc = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(precomp),&precomp);
            infoVT.back().buffer.offset = 0u;
            infoVT.back().buffer.size = sizeof(video::IGPUVirtualTexture::SPrecomputedData);
            writesVT.back().dstSet = sceneData.vtDS.get();
            writesVT.back().binding = 2u;
            writesVT.back().arrayElement = 0u;
            writesVT.back().count = 1u;
            writesVT.back().descriptorType = EDT_STORAGE_BUFFER;
            writesVT.back().info = &infoVT.back();

            driver->updateDescriptorSets(writesVT.size(),writesVT.data(),0u,nullptr);
        }
        smart_refctd_ptr<IGPUDescriptorSetLayout> vgDSLayout;
        std::string extraCode;
        // msvc is incredibly dumb and complains about type mismatches in code sections guarded by if constexpr
        std::conditional_t<useSSBO,GPUMeshPacker::DSLayoutParamsSSBO,GPUMeshPacker::DSLayoutParamsUTB> tmp;
        [&](auto& layoutParams) -> void
        {
            // layout
            core::vector<IGPUDescriptorSetLayout::SBinding> vgBindings((useSSBO ? gpump->getDSlayoutBindingsForSSBO(nullptr):gpump->getDSlayoutBindingsForUTB(nullptr))+1u);
            auto& materialDataBinding = vgBindings.front();
            materialDataBinding.binding = 0u;
            materialDataBinding.type = EDT_STORAGE_BUFFER;
            materialDataBinding.count = 1u;
            materialDataBinding.samplers = nullptr; // not sampler interpolated
            auto* actualVGBindings = vgBindings.data()+1u;
            if constexpr (useSSBO)
            {
                layoutParams.uintBufferBinding = 1u;
                layoutParams.uvec2BufferBinding = 2u;
                layoutParams.uvec3BufferBinding = 3u;
                layoutParams.uvec4BufferBinding = 4u;
                layoutParams.indexBufferBinding = 5u;
                gpump->getDSlayoutBindingsForSSBO(actualVGBindings,layoutParams);
            }
            else
            {
                layoutParams.usamplersBinding = 1u;
                layoutParams.fsamplersBinding = 2u;
                gpump->getDSlayoutBindingsForUTB(actualVGBindings,layoutParams);
            }
            for (auto& binding : vgBindings)
                binding.stageFlags = static_cast<ISpecializedShader::E_SHADER_STAGE>(ISpecializedShader::ESS_VERTEX|ISpecializedShader::ESS_COMPUTE|ISpecializedShader::ESS_FRAGMENT);

            vgDSLayout = driver->createGPUDescriptorSetLayout(vgBindings.data(),vgBindings.data()+vgBindings.size());

            // write
            sceneData.vgDS = driver->createGPUDescriptorSet(smart_refctd_ptr(vgDSLayout));

            uint32_t writeCount,infoCount;
            if constexpr (useSSBO)
            {
                writeCount = gpump->getDescriptorSetWritesForSSBO(nullptr,nullptr,nullptr);
                infoCount = 2u;
            }
            else
                std::tie(writeCount,infoCount) = gpump->getDescriptorSetWritesForUTB(nullptr,nullptr,nullptr);
            vector<IGPUDescriptorSet::SWriteDescriptorSet> writesVG(++writeCount);
            vector<IGPUDescriptorSet::SDescriptorInfo> infosVG(++infoCount);

            auto writes = writesVG.data();
            auto infos = infosVG.data();
            writes->dstSet = sceneData.vgDS.get();
            writes->binding = 0u;
            writes->arrayElement = 0u;
            writes->count = 1u;
            writes->descriptorType = EDT_STORAGE_BUFFER;
            writes->info = infos;
            writes++;
            infos->buffer.offset = 0u;
            infos->buffer.size = batchDataSSBO->getSize();
            infos->desc = std::move(batchDataSSBO);
            infos++;

            constexpr uint32_t vgDescriptorSetIx = 1u;
            if constexpr (useSSBO)
            {
                extraCode = gpump->getGLSLForSSBO(vgDescriptorSetIx,layoutParams);
                gpump->getDescriptorSetWritesForSSBO(writes,infos,sceneData.vgDS.get(),layoutParams);
            }
            else
            {
                extraCode = gpump->getGLSLForUTB(vgDescriptorSetIx, layoutParams);
                gpump->getDescriptorSetWritesForUTB(writes,infos,sceneData.vgDS.get(),layoutParams);
            }
            driver->updateDescriptorSets(writeCount,writesVG.data(),0u,nullptr);
        }(tmp);
        auto overrideShaderJustAfterVersionDirective = [am,driver,&extraCode](const char* path)
        {
            asset::IAssetLoader::SAssetLoadParams lp;
            auto _specShader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset(path,lp).getContents().begin());
            assert(_specShader);
            const asset::ICPUShader* unspec = _specShader->getUnspecialized();
            assert(unspec->containsGLSL());

            auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
            const std::string_view origSource(begin, unspec->getSPVorGLSL()->getSize());

            const size_t firstNewlineAfterVersion = origSource.find("\n", origSource.find("#version "));
            assert(firstNewlineAfterVersion != std::string_view::npos);
            const std::string_view sourceWithoutVersion(begin + firstNewlineAfterVersion, origSource.size() - firstNewlineAfterVersion);

            std::string newSource("#version 460 core\n");
            newSource += extraCode;
            newSource += sourceWithoutVersion;

            auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(newSource.c_str());
            auto specinfo = _specShader->getSpecializationInfo();

            auto newSpecShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));
            auto gpuSpecShaders = driver->getGPUObjectsFromAssets(&newSpecShader.get(), &newSpecShader.get() + 1u);
            return std::move(gpuSpecShaders->begin()[0]);
        };
        //
        {
            SPushConstantRange pcRange;
            pcRange.size = sizeof(uint32_t);
            pcRange.offset = 0u;
            pcRange.stageFlags = ISpecializedShader::ESS_VERTEX;

            smart_refctd_ptr<IGPUSpecializedShader> fillShaders[2] = {
                overrideShaderJustAfterVersionDirective("../fillVBuffer.vert"),
                overrideShaderJustAfterVersionDirective("../fillVBuffer.frag")
            };

            sceneData.fillVBufferPpln = driver->createGPURenderpassIndependentPipeline(
                nullptr,driver->createGPUPipelineLayout(&pcRange,&pcRange+1,smart_refctd_ptr(vtDSLayout),smart_refctd_ptr(vgDSLayout),smart_refctd_ptr(perFrameDSLayout)),
                &fillShaders[0].get(),&fillShaders[0].get()+2u,
                SVertexInputParams{},
                SBlendParams{},
                SPrimitiveAssemblyParams{},
                SRasterizationParams{}
            );
        }
        {
            extraCode += "#define _NBL_VT_FLOAT_VIEWS_COUNT "+std::to_string(gpuvt->getFloatViews().size())+"\n";
            std::cout << gpuvt->getGLSLFunctionsIncludePath().c_str() << std::endl;

            SPushConstantRange pcRange;
            pcRange.size = sizeof(core::vector3df);
            pcRange.offset = 0u;
            pcRange.stageFlags = ISpecializedShader::ESS_COMPUTE;

            sceneData.shadeVBufferPpln = driver->createGPUComputePipeline(
                nullptr,driver->createGPUPipelineLayout(&pcRange,&pcRange+1,std::move(vtDSLayout),std::move(vgDSLayout),std::move(perFrameDSLayout),std::move(shadingDSLayout)),
                overrideShaderJustAfterVersionDirective("../shadeVBuffer.comp")
            );
        }
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);
    

    uint64_t lastFPSTime = 0;
    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        SBasicViewParameters uboData;
        memcpy(uboData.MVP, camera->getConcatenatedMatrix().pointer(), sizeof(core::matrix4SIMD));
        memcpy(uboData.MV, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));
        memcpy(uboData.NormalMat, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));
        driver->updateBufferRangeViaStagingBuffer(sceneData.ubo.get(), 0u, sizeof(SBasicViewParameters), &uboData);

        // TODO: Cull MDIs

        driver->setRenderTarget(visBuffer);
        driver->clearZBuffer();
        const uint32_t invalidObjectCode[4] = {~0u,0u,0u,0u};
        driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0,invalidObjectCode);

        const IGPUDescriptorSet* ds[4] = {sceneData.vtDS.get(),sceneData.vgDS.get(),sceneData.perFrameDS.get(),sceneData.shadingDS.get()};
        driver->bindDescriptorSets(video::EPBP_GRAPHICS,sceneData.fillVBufferPpln->getLayout(),0u,3u,ds,nullptr);
        // fill visibility buffer
        driver->bindGraphicsPipeline(sceneData.fillVBufferPpln.get());
        for (auto i = 0u; i<sceneData.pushConstantsData.size(); i++)
        {
            driver->pushConstants(sceneData.fillVBufferPpln->getLayout(),IGPUSpecializedShader::ESS_ALL,0u,sizeof(uint32_t),sceneData.pushConstantsData.data()+i);

            const asset::SBufferBinding<IGPUBuffer> noVtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
            driver->drawIndexedIndirect(
                noVtxBindings,DrawIndexedIndirectInput::mode,DrawIndexedIndirectInput::indexType,
                sceneData.idxBuffer.get(),sceneData.mdiBuffer.get(),
                sceneData.drawIndirectInput[i].offset,sceneData.drawIndirectInput[i].maxCount,
                sizeof(DrawElementsIndirectCommand_t)
            );
        }

        // shade
        driver->bindDescriptorSets(video::EPBP_COMPUTE,sceneData.shadeVBufferPpln->getLayout(),0u,4u,ds,nullptr);
        driver->bindComputePipeline(sceneData.shadeVBufferPpln.get());
        {
            auto camPos = camera->getAbsolutePosition();
            driver->pushConstants(sceneData.shadeVBufferPpln->getLayout(),IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(camPos),&camPos.X);
        }
        driver->dispatch((params.WindowSize.Width-1u)/SHADING_WG_SIZE_X+1u,(params.WindowSize.Height-1u)/SHADING_WG_SIZE_Y+1u,1u);
        COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        // blit
        driver->blitRenderTargets(fb,0);
        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream str;
            str << L"Visibility Buffer - Nabla Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(str.str().c_str());
            lastFPSTime = time;
        }
    }
    driver->removeAllFrameBuffers();
}