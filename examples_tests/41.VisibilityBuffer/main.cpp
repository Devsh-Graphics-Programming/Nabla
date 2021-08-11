// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "nbl/ext/DepthPyramidGenerator/DepthPyramidGenerator.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

bool freezeCulling = false;

class MyEventReceiver : public QToQuitEventReceiver
{
public:

    MyEventReceiver()
    {
    }

    bool OnEvent(const SEvent& event)
    {
        if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case nbl::KEY_KEY_Q: // so we can quit
                return QToQuitEventReceiver::OnEvent(event);
            case nbl::KEY_KEY_C: // freeze culling
                freezeCulling = !freezeCulling; // Not enabled/necessary yet
                return true;
            default:
                break;
            }
        }

        return false;
    }
};

#include "common.h"
#include "rasterizationCommon.h"

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

    core::vector<CullData_t> cullData;
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

    smart_refctd_ptr<IGPUBuffer> idxBuffer;
    //TODO: mdi buffers should be in the `cullShaderData` struct?
    smart_refctd_ptr<IGPUBuffer> frustumCulledMdiBuffer;
    smart_refctd_ptr<IGPUBuffer> occlusionCulledMdiBuffer;
    smart_refctd_ptr<IGPUDescriptorSet> vtDS,vgDS,perFrameDS,shadingDS;

    core::vector<DrawIndexedIndirectInput> drawIndirectInput;
    core::vector<uint32_t> pushConstantsData;

    smart_refctd_ptr<IGPUBuffer> ubo;

};

//TODO: split into two structs (one for frustrum culling, one for occlusion)
struct CullShaderData
{
    core::smart_refctd_ptr<IGPUBuffer> perBatchCull;
    core::smart_refctd_ptr<IGPUBuffer> mvpBuffer;

    // frustum culling
    core::smart_refctd_ptr<IGPUComputePipeline> cullPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> cullDSLayout;
    core::smart_refctd_ptr<IGPUDescriptorSet> cullDS;

    // occlusion culling
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> occlusionCullPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> occlusionCullDSLayout;
    core::smart_refctd_ptr<IGPUDescriptorSet> occlusionCullDS;

    // batch ID to MDI offset mapping
    core::smart_refctd_ptr<IGPUComputePipeline> mapPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> mapDSLayout;
    core::smart_refctd_ptr<IGPUDescriptorSet> mapDS;

    SBufferBinding<IGPUBuffer> cubeIdxBuffer;
    SBufferBinding<IGPUBuffer> cubeVertexBuffers[SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    core::smart_refctd_ptr<IGPUBuffer> visible;
    core::smart_refctd_ptr<IGPUBuffer> cubeMVPs;
    core::smart_refctd_ptr<IGPUBuffer> cubeDrawGUIDs;
    core::smart_refctd_ptr<IGPUBuffer> cubeCommandBuffer;
    core::smart_refctd_ptr<IGPUBuffer> dispatchIndirect;

    uint32_t maxBatchCount;
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
    MyEventReceiver receiver;
    device->setEventReceiver(&receiver);

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* fs = am->getFileSystem();

    auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(driver);

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

    auto zBuffOnlyFrameBuffer = driver->addFrameBuffer();
    zBuffOnlyFrameBuffer->attach(EFAP_DEPTH_ATTACHMENT, std::move(depthBufferView));

    //
    SceneData sceneData;
    CullShaderData cullShaderData;
    core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> dbgLines;
    {
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
            allocParams.indexBuffSupportedCnt = 32u * 1024u * 1024u;
            allocParams.indexBufferMinAllocCnt = minTrisBatch*kVerticesPerTriangle;
            allocParams.vertexBuffSupportedByteSize = 128u * 1024u * 1024u;
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

            core::vector<CullData_t> batchCullData(mdiCntBound);
            auto batchCullDataEnd = batchCullData.begin();

            allocDataIt = allocData->begin();
            uint32_t mdiListOffset = 0u;
            for (auto it=pipelineMeshBufferRanges.begin(); it!=pipelineMeshBufferRanges.end()-1u; )
            {
                auto mbRangeBegin = &(*it)->get();
                auto mbRangeEnd = &(*(++it))->get();

                const uint32_t meshMdiBound = mp->calcMDIStructMaxCount(mbRangeBegin,mbRangeEnd);
                core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(std::distance(mbRangeBegin,mbRangeEnd));
                core::vector<MeshPacker::CombinedDataOffsetTable> cdot(meshMdiBound);
                core::vector<aabbox3df> aabbs(meshMdiBound);
                uint32_t actualMdiCnt = mp->commit(pmbd.data(),cdot.data(),aabbs.data(),&*allocDataIt,mbRangeBegin,mbRangeEnd);
                allocDataIt += meshMdiBound;

                if (actualMdiCnt==0u)
                {
                    std::cout << "Commit failed \n";
                    _NBL_DEBUG_BREAK_IF(true);
                }

                uint32_t aabbIdx = 0u;
                for (auto packedMeshBufferData : pmbd)
                {
                    for (uint32_t i = 0u; i < packedMeshBufferData.mdiParameterCount; i++)
                    {
                        batchCullDataEnd->aabbMinEdge.x = aabbs[aabbIdx].MinEdge.X;
                        batchCullDataEnd->aabbMinEdge.y = aabbs[aabbIdx].MinEdge.Y;
                        batchCullDataEnd->aabbMinEdge.z = aabbs[aabbIdx].MinEdge.Z;

                        batchCullDataEnd->aabbMaxEdge.x = aabbs[aabbIdx].MaxEdge.X;
                        batchCullDataEnd->aabbMaxEdge.y = aabbs[aabbIdx].MaxEdge.Y;
                        batchCullDataEnd->aabbMaxEdge.z = aabbs[aabbIdx].MaxEdge.Z;

                        batchCullDataEnd->drawCommandGUID = packedMeshBufferData.mdiParameterOffset + i;

                        draw3DLine->enqueueBox(dbgLines, aabbs[aabbIdx], 0.0f, 0.0f, 0.0f, 1.0f, core::matrix3x4SIMD());

                        batchCullDataEnd++;
                        aabbIdx++;
                    }
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

            const uint32_t totalActualMdiCnt = mdiListOffset;

            batchDataSSBO = driver->createFilledDeviceLocalGPUBufferOnDedMem(batchData.size()*sizeof(BatchInstanceData),batchData.data());

            gpump = core::make_smart_refctd_ptr<CGPUMeshPackerV2<>>(driver,mp.get());
            sceneData.idxBuffer = gpump->getPackerDataStore().indexBuffer;

            sceneData.frustumCulledMdiBuffer = gpump->getPackerDataStore().MDIDataBuffer;
            sceneData.occlusionCulledMdiBuffer = driver->createDeviceLocalGPUBufferOnDedMem(sceneData.frustumCulledMdiBuffer->getSize());
            driver->copyBuffer(sceneData.frustumCulledMdiBuffer.get(), sceneData.occlusionCulledMdiBuffer.get(), 0u, 0u, sceneData.frustumCulledMdiBuffer->getSize());

            cullShaderData.cubeMVPs = driver->createDeviceLocalGPUBufferOnDedMem(totalActualMdiCnt * sizeof(core::matrix4SIMD));
            cullShaderData.cubeDrawGUIDs = driver->createDeviceLocalGPUBufferOnDedMem(totalActualMdiCnt * sizeof(uint32_t));

            cullShaderData.maxBatchCount = std::distance(batchCullData.begin(), batchCullDataEnd);
            cullShaderData.perBatchCull = driver->createFilledDeviceLocalGPUBufferOnDedMem(cullShaderData.maxBatchCount * sizeof(CullData_t), batchCullData.data());
            cullShaderData.mvpBuffer = driver->createDeviceLocalGPUBufferOnDedMem(cullShaderData.maxBatchCount * sizeof(core::matrix4SIMD));
            cullShaderData.visible = driver->createDeviceLocalGPUBufferOnDedMem(totalActualMdiCnt * sizeof(uint16_t));
            driver->fillBuffer(cullShaderData.visible.get(), 0u, cullShaderData.visible->getSize(), 0u);
        }
        mesh_raw->convertToDummyObject(~0u);

        // 
        {
            constexpr uint32_t cubeIdxCnt = 36u;

            DrawElementsIndirectCommand_t cubeIndirectDrawCommand;
            cubeIndirectDrawCommand.baseInstance = 0u;
            cubeIndirectDrawCommand.baseVertex = 0u;
            cubeIndirectDrawCommand.count = cubeIdxCnt;
            cubeIndirectDrawCommand.firstIndex = 0u;
            cubeIndirectDrawCommand.instanceCount = 0u;
            cullShaderData.cubeCommandBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(DrawElementsIndirectCommand_t), &cubeIndirectDrawCommand);

            DispatchIndirectCommand_t dispatchIndirect;
            dispatchIndirect.num_groups_x = 0u;
            dispatchIndirect.num_groups_y = 1u;
            dispatchIndirect.num_groups_z = 1u;
            cullShaderData.dispatchIndirect = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(DispatchIndirectCommand_t), &dispatchIndirect);
        }

        //
        smart_refctd_ptr<IGPUDescriptorSetLayout> perFrameDSLayout, shadingDSLayout;
        {
            {
                IGPUDescriptorSetLayout::SBinding bindings[1];
                bindings[0].binding = 0u;
                bindings[0].count = 1u;
                bindings[0].samplers = nullptr;
                bindings[0].stageFlags = ISpecializedShader::ESS_VERTEX;
                bindings[0].type = EDT_UNIFORM_BUFFER;

                perFrameDSLayout = driver->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
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
                writes[0].info = infos + 0u;
                driver->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
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

                IGPUDescriptorSetLayout::SBinding bindings[5];
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

                for (uint32_t i = 2u; i < 5u; i++)
                {
                    bindings[i].binding = i;
                    bindings[i].count = 1u;
                    bindings[i].samplers = nullptr;
                    bindings[i].stageFlags = ISpecializedShader::ESS_COMPUTE;
                    bindings[i].type = EDT_STORAGE_BUFFER;
                }

                shadingDSLayout = driver->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
            }
            {
                IGPUDescriptorSet::SDescriptorInfo infos[5];
                infos[0].desc = core::smart_refctd_ptr(visBufferView);
                //infos[0].image.imageLayout = ?;
                infos[0].image.sampler = nullptr; // used immutable in the layout
                infos[1].desc = createImageView(std::move(framebuffer), EF_R8G8B8A8_UNORM);
                //infos[0].image.imageLayout = ?;
                infos[1].image.sampler = nullptr; // storage image

                infos[2].buffer.offset = 0u;
                infos[2].buffer.size = cullShaderData.visible->getSize();
                infos[2].desc = cullShaderData.visible;

                infos[3].buffer.offset = 0u;
                infos[3].buffer.size = cullShaderData.cubeCommandBuffer->getSize();
                infos[3].desc = cullShaderData.cubeCommandBuffer;

                infos[4].buffer.offset = 0u;
                infos[4].buffer.size = cullShaderData.dispatchIndirect->getSize();
                infos[4].desc = cullShaderData.dispatchIndirect;

                sceneData.shadingDS = driver->createGPUDescriptorSet(smart_refctd_ptr(shadingDSLayout));
                IGPUDescriptorSet::SWriteDescriptorSet writes[5];
                for (auto i = 0u; i < 5u; i++)
                {
                    writes[i].dstSet = sceneData.shadingDS.get();
                    writes[i].binding = i;
                    writes[i].arrayElement = 0u;
                    writes[i].count = 1u;
                    writes[i].info = infos + i;
                }
                writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
                writes[1].descriptorType = EDT_STORAGE_IMAGE;
                writes[2].descriptorType = EDT_STORAGE_BUFFER;
                writes[3].descriptorType = EDT_STORAGE_BUFFER;
                writes[4].descriptorType = EDT_STORAGE_BUFFER;

                driver->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
            }
        }

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
        // msvc is incredibly dumb and complains about type mismatches in code sections guarded by ifconstexpr
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
                writeCount = gpump->getDescriptorSetWritesForSSBO(nullptr, nullptr, nullptr);
                infoCount = 2u;
            }
            else
            std::tie(writeCount, infoCount) = gpump->getDescriptorSetWritesForUTB(nullptr, nullptr, nullptr);
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
                extraCode = gpump->getGLSLForSSBO(vgDescriptorSetIx, layoutParams);
                gpump->getDescriptorSetWritesForSSBO(writes, infos, sceneData.vgDS.get(), layoutParams);
            }
            else
            {
                extraCode = gpump->getGLSLForUTB(vgDescriptorSetIx, layoutParams);
                gpump->getDescriptorSetWritesForUTB(writes, infos, sceneData.vgDS.get(), layoutParams);
            }
            driver->updateDescriptorSets(writeCount, writesVG.data(), 0u, nullptr);
        }(tmp);
        auto overrideShaderJustAfterVersionDirective = [am, driver, &extraCode](const char* path)
        {
            asset::IAssetLoader::SAssetLoadParams lp;
            auto _specShader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset(path, lp).getContents().begin());
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
                nullptr, driver->createGPUPipelineLayout(&pcRange, &pcRange + 1, smart_refctd_ptr(vtDSLayout), smart_refctd_ptr(vgDSLayout), smart_refctd_ptr(perFrameDSLayout)),
                &fillShaders[0].get(), &fillShaders[0].get() + 2u,
                SVertexInputParams{},
                SBlendParams{},
                SPrimitiveAssemblyParams{},
                SRasterizationParams{}
            );
        }
        {
            extraCode += "#define _NBL_VT_FLOAT_VIEWS_COUNT " + std::to_string(gpuvt->getFloatViews().size()) + "\n";
            std::cout << gpuvt->getGLSLFunctionsIncludePath().c_str() << std::endl;

            SPushConstantRange pcRange;
            pcRange.size = sizeof(core::vector3df);
            pcRange.offset = 0u;
            pcRange.stageFlags = ISpecializedShader::ESS_COMPUTE;

            sceneData.shadeVBufferPpln = driver->createGPUComputePipeline(
                nullptr, driver->createGPUPipelineLayout(&pcRange, &pcRange + 1, std::move(vtDSLayout), std::move(vgDSLayout), std::move(perFrameDSLayout), std::move(shadingDSLayout)),
                overrideShaderJustAfterVersionDirective("../shadeVBuffer.comp")
            );
        }
    }

    // occlusion cull shader pipeline setup
    {
        cullShaderData.cubeVertexBuffers[0].offset = 0ull;
        cullShaderData.cubeVertexBuffers[0].buffer = cullShaderData.cubeMVPs;
        
        const uint16_t indices[36] = {
            0, 2, 1,
            0, 3, 2,
            0, 4, 3,
            3, 4, 7,
            4, 5, 7,
            5, 6, 7,
            5, 1, 6,
            1, 2, 6,
            7, 6, 3,
            6, 2, 3,
            4, 0, 5,
            0, 1, 5
        };

        cullShaderData.cubeIdxBuffer.offset = 0ull;
        cullShaderData.cubeIdxBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(indices), indices);

        SVertexInputParams vtxParams;
        vtxParams.enabledAttribFlags = 0b1111u;
        vtxParams.enabledBindingFlags = 0b1u;

        vtxParams.bindings[0].inputRate = EVIR_PER_INSTANCE;
        vtxParams.bindings[0].stride = sizeof(core::vectorSIMDf) * 4u;

        for (uint32_t i = 0u; i < 4u; i++)
        {
            vtxParams.attributes[i].binding = 0u;
            vtxParams.attributes[i].format = EF_R32G32B32A32_SFLOAT;
            vtxParams.attributes[i].relativeOffset = sizeof(core::vectorSIMDf) * i;
        }

        {
            IGPUDescriptorSetLayout::SBinding bindings[1];

            bindings[0].binding = 0u;
            bindings[0].count = 1u;
            bindings[0].samplers = nullptr;
            bindings[0].stageFlags = ISpecializedShader::ESS_FRAGMENT;
            bindings[0].type = EDT_STORAGE_BUFFER;

            cullShaderData.occlusionCullDSLayout = driver->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
        }

        {
            cullShaderData.occlusionCullDS = driver->createGPUDescriptorSet(smart_refctd_ptr(cullShaderData.occlusionCullDSLayout));

            IGPUDescriptorSet::SDescriptorInfo info[1];
            info[0].desc = core::smart_refctd_ptr(cullShaderData.visible);
            info[0].buffer.offset = 0u;
            info[0].buffer.size = cullShaderData.visible->getSize();


            IGPUDescriptorSet::SWriteDescriptorSet write[1];
            write[0].dstSet = cullShaderData.occlusionCullDS.get();
            write[0].binding = 0;
            write[0].arrayElement = 0u;
            write[0].count = 1u;
            write[0].descriptorType = EDT_STORAGE_BUFFER;
            write[0].info = info;

            driver->updateDescriptorSets(sizeof(write) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), write, 0u, nullptr);
        }

        asset::IAssetLoader::SAssetLoadParams lp;
        auto vtxOcclusionShader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset("../occlusionCull.vert", lp).getContents().begin());
        auto fragOcclusionShader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset("../occlusionCull.frag", lp).getContents().begin());
        assert(vtxOcclusionShader);
        assert(fragOcclusionShader);
        assert(vtxOcclusionShader->getUnspecialized()->containsGLSL());
        assert(fragOcclusionShader->getUnspecialized()->containsGLSL());

        auto gpuVtxOcclusionShader = driver->getGPUObjectsFromAssets(&vtxOcclusionShader, &vtxOcclusionShader + 1u)->begin()[0];
        auto gpuFragOcclusionShader = driver->getGPUObjectsFromAssets(&fragOcclusionShader, &fragOcclusionShader + 1u)->begin()[0];

        IGPUSpecializedShader* occlusionShaders[2] = {
            gpuVtxOcclusionShader.get(), gpuFragOcclusionShader.get()
        };

        SRasterizationParams rasterizationParams;
        rasterizationParams.depthWriteEnable = false;

        cullShaderData.occlusionCullPipeline = driver->createGPURenderpassIndependentPipeline(
            nullptr, driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(cullShaderData.occlusionCullDSLayout)),
            occlusionShaders, occlusionShaders + 2u,
            vtxParams,
            SBlendParams{},
            SPrimitiveAssemblyParams{},
            rasterizationParams
        );
    }

    // map shader pipeline setup
    {
        SPushConstantRange range{ ISpecializedShader::ESS_COMPUTE,0u,sizeof(uint32_t) };

        {
            IGPUDescriptorSetLayout::SBinding bindings[4];
            for (uint32_t i = 0u; i < 4; i++)
            {
                bindings[i].binding = i;
                bindings[i].count = 1u;
                bindings[i].samplers = nullptr;
                bindings[i].stageFlags = ISpecializedShader::ESS_COMPUTE;
                bindings[i].type = EDT_STORAGE_BUFFER;
            }

            cullShaderData.mapDSLayout = driver->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
        }
        
        {
            IGPUDescriptorSet::SDescriptorInfo infos[4];

            infos[0].desc = sceneData.occlusionCulledMdiBuffer;
            infos[0].buffer.offset = 0u;
            infos[0].buffer.size = sceneData.occlusionCulledMdiBuffer->getSize();

            infos[1].desc = cullShaderData.visible;
            infos[1].buffer.offset = 0u;
            infos[1].buffer.size = cullShaderData.visible->getSize();

            infos[2].desc = cullShaderData.cubeDrawGUIDs;
            infos[2].buffer.offset = 0u;
            infos[2].buffer.size = cullShaderData.cubeDrawGUIDs->getSize();

            infos[3].desc = cullShaderData.cubeCommandBuffer;
            infos[3].buffer.offset = 0u;
            infos[3].buffer.size = cullShaderData.cubeCommandBuffer->getSize();

            cullShaderData.mapDS = driver->createGPUDescriptorSet(smart_refctd_ptr(cullShaderData.mapDSLayout));

            IGPUDescriptorSet::SWriteDescriptorSet writes[4];
            for (uint32_t i = 0u; i < 4; i++)
            {
                writes[i].dstSet = cullShaderData.mapDS.get();
                writes[i].binding = i;
                writes[i].arrayElement = 0u;
                writes[i].count = 1u;
                writes[i].descriptorType = EDT_STORAGE_BUFFER;
                writes[i].info = infos + i;
            }

            driver->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
        }

        asset::IAssetLoader::SAssetLoadParams lp;
        auto mapShader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset("../occlusionCullMap.comp", lp).getContents().begin());
        assert(mapShader);
        const asset::ICPUShader* unspec = mapShader->getUnspecialized();
        assert(unspec->containsGLSL());

        auto gpuMapShader = driver->getGPUObjectsFromAssets(&mapShader, &mapShader + 1u)->begin()[0];

        auto mapPipelineLayout = driver->createGPUPipelineLayout(&range, &range + 1u, core::smart_refctd_ptr(cullShaderData.mapDSLayout));
        cullShaderData.mapPipeline = driver->createGPUComputePipeline(nullptr, std::move(mapPipelineLayout), std::move(gpuMapShader));
    }

    // frustum cull shader pipeline setup
    {
        SPushConstantRange range{ ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t) };
        
        {
            IGPUDescriptorSetLayout::SBinding bindings[9];
            for (uint32_t i = 0u; i < 9; i++)
            {
                bindings[i].binding = i;
                bindings[i].count = 1u;
                bindings[i].samplers = nullptr;
                bindings[i].stageFlags = ISpecializedShader::ESS_COMPUTE;
                bindings[i].type = EDT_STORAGE_BUFFER;
            }

            cullShaderData.cullDSLayout = driver->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
        }
        
        {
            IGPUDescriptorSet::SDescriptorInfo infos[9];
        
            infos[0].desc = core::smart_refctd_ptr(cullShaderData.perBatchCull);
            infos[0].buffer.offset = 0u;
            infos[0].buffer.size = cullShaderData.perBatchCull->getSize();
        
            infos[1].desc = core::smart_refctd_ptr(cullShaderData.mvpBuffer);
            infos[1].buffer.offset = 0u;
            infos[1].buffer.size = cullShaderData.mvpBuffer->getSize();

            infos[2].desc = cullShaderData.cubeVertexBuffers[0].buffer;
            infos[2].buffer.offset = 0u;
            infos[2].buffer.size = cullShaderData.cubeVertexBuffers[0].buffer->getSize();

            infos[3].desc = cullShaderData.cubeCommandBuffer;
            infos[3].buffer.offset = 0u;
            infos[3].buffer.size = cullShaderData.cubeCommandBuffer->getSize();

            infos[4].desc = sceneData.frustumCulledMdiBuffer;
            infos[4].buffer.offset = 0u;
            infos[4].buffer.size = sceneData.frustumCulledMdiBuffer->getSize();

            infos[5].desc = sceneData.occlusionCulledMdiBuffer;
            infos[5].buffer.offset = 0u;
            infos[5].buffer.size = sceneData.occlusionCulledMdiBuffer->getSize();

            infos[6].desc = cullShaderData.cubeDrawGUIDs;
            infos[6].buffer.offset = 0u;
            infos[6].buffer.size = cullShaderData.cubeDrawGUIDs->getSize();

            infos[7].desc = cullShaderData.dispatchIndirect;
            infos[7].buffer.offset = 0u;
            infos[7].buffer.size = cullShaderData.dispatchIndirect->getSize();

            infos[8].desc = cullShaderData.visible;
            infos[8].buffer.offset = 0u;
            infos[8].buffer.size = cullShaderData.visible->getSize();

            cullShaderData.cullDS = driver->createGPUDescriptorSet(smart_refctd_ptr(cullShaderData.cullDSLayout));
        
            IGPUDescriptorSet::SWriteDescriptorSet writes[9];
            for (uint32_t i = 0u; i < 9; i++)
            {
                writes[i].dstSet = cullShaderData.cullDS.get();
                writes[i].binding = i;
                writes[i].arrayElement = 0u;
                writes[i].count = 1u;
                writes[i].descriptorType = EDT_STORAGE_BUFFER;
                writes[i].info = infos + i;
            }
        
            driver->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
        }
        
        asset::IAssetLoader::SAssetLoadParams lp;
        auto cullShader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset("../cull.comp", lp).getContents().begin());
        assert(cullShader);
        const asset::ICPUShader* unspec = cullShader->getUnspecialized();
        assert(unspec->containsGLSL());
        
        auto gpuCullShader = driver->getGPUObjectsFromAssets(&cullShader, &cullShader + 1u)->begin()[0];
        
        auto cullPipelineLayout = driver->createGPUPipelineLayout(&range, &range + 1u, core::smart_refctd_ptr(cullShaderData.cullDSLayout));
        cullShaderData.cullPipeline = driver->createGPUComputePipeline(nullptr, std::move(cullPipelineLayout), std::move(gpuCullShader));
    }

    auto cullBatches = [&driver, &cullShaderData, &sceneData](const core::matrix4SIMD& vp, const core::vector3df& camPos, bool freezeCulling)
    {
        driver->bindDescriptorSets(EPBP_COMPUTE, cullShaderData.cullPipeline->getLayout(), 0u, 1u, &cullShaderData.cullDS.get(), nullptr);
        driver->bindComputePipeline(cullShaderData.cullPipeline.get());

        CullShaderData_t cullPushConstants;
        cullPushConstants.viewProjMatrix = vp;
        cullPushConstants.worldCamPos.x = camPos.X;
        cullPushConstants.worldCamPos.y = camPos.Y;
        cullPushConstants.worldCamPos.z = camPos.Z;
        cullPushConstants.freezeCullingAndMaxBatchCountPacked = cullShaderData.maxBatchCount | (static_cast<uint32_t>(freezeCulling) << 16u);

        driver->pushConstants(cullShaderData.cullPipeline->getLayout(), ISpecializedShader::ESS_COMPUTE, 0u, sizeof(CullShaderData_t), &cullPushConstants);

        const uint32_t cullWorkGroups = (cullShaderData.maxBatchCount - 1u) / WORKGROUP_SIZE + 1u;

        driver->dispatch(cullWorkGroups, 1u, 1u);
    };

    const IGPUDescriptorSet* vBuffDs[4] = { sceneData.vtDS.get(),sceneData.vgDS.get(),sceneData.perFrameDS.get(),sceneData.shadingDS.get() };
    auto fillVBuffer = [&driver, &sceneData, &vBuffDs](core::smart_refctd_ptr<IGPUBuffer> mdiBuffer)
    {
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, sceneData.fillVBufferPpln->getLayout(), 0u, 3u, vBuffDs, nullptr);
        driver->bindGraphicsPipeline(sceneData.fillVBufferPpln.get());

        for (auto i = 0u; i < sceneData.pushConstantsData.size(); i++)
        {
            driver->pushConstants(sceneData.fillVBufferPpln->getLayout(), IGPUSpecializedShader::ESS_ALL, 0u, sizeof(uint32_t), &sceneData.pushConstantsData[i]);

            const asset::SBufferBinding<IGPUBuffer> noVtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
            driver->drawIndexedIndirect(
                noVtxBindings, DrawIndexedIndirectInput::mode, DrawIndexedIndirectInput::indexType,
                sceneData.idxBuffer.get(), mdiBuffer.get(),
                sceneData.drawIndirectInput[i].offset, sceneData.drawIndirectInput[i].maxCount,
                sizeof(DrawElementsIndirectCommand_t)
            );
        }
    };

    using DPG = ext::DepthPyramidGenerator::DepthPyramidGenerator;

    DPG::Config config;
    DPG dpg(driver, am, depthBufferView, config);

    const uint32_t mipCnt = DPG::getMaxMipCntFromImage(depthBufferView);
    core::vector<core::smart_refctd_ptr<IGPUImageView>> mips(mipCnt);
    DPG::createMipMapImageViews(driver, depthBufferView, mips.data(), config);

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> dpgDsLayout;
    const uint32_t dpgDsCnt = DPG::createDescriptorSets(driver, depthBufferView, mips.data(), dpgDsLayout, nullptr, nullptr, config);
    core::vector<core::smart_refctd_ptr<IGPUDescriptorSet>> dpgDs(dpgDsCnt);
    core::vector<uint32_t> dpgPushConstants(dpgDsCnt);
    DPG::createDescriptorSets(driver, depthBufferView, mips.data(), dpgDsLayout, dpgDs.data(), dpgPushConstants.data(), config);

    core::smart_refctd_ptr<IGPUComputePipeline> dpgPpln;
    dpg.createPipeline(driver, dpgDsLayout, dpgPpln, config);

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    bool asdf = true;
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

        // frustum cull
        // TODO: fill instanceCounts of frustumCulledMdiBuffer with zeros
        cullBatches(camera->getConcatenatedMatrix(), camera->getPosition(), freezeCulling);
        COpenGLExtensionHandler::pGlMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        // first fill visibility buffer pass
        driver->setRenderTarget(visBuffer);
        driver->clearZBuffer();
        const uint32_t invalidObjectCode[4] = { ~0u,0u,0u,0u };
        driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, invalidObjectCode);
        fillVBuffer(sceneData.frustumCulledMdiBuffer);

        // create depth pyramid
        for (uint32_t i = 0u; i < dpgDsCnt; i++)
            dpg.generateMipMaps(depthBufferView, dpgPpln, dpgDs[i], dpgPushConstants[i]);

        // occlusion cull (against partially filled new Z-buffer)
        driver->setRenderTarget(zBuffOnlyFrameBuffer);
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, cullShaderData.occlusionCullPipeline->getLayout(), 0u, 1u, &cullShaderData.occlusionCullDS.get(), nullptr);
        driver->bindGraphicsPipeline(cullShaderData.occlusionCullPipeline.get());

        driver->drawIndexedIndirect(
            cullShaderData.cubeVertexBuffers, EPT_TRIANGLE_LIST, EIT_16BIT,
            cullShaderData.cubeIdxBuffer.buffer.get(), cullShaderData.cubeCommandBuffer.get(),
            0u, 1u,
            sizeof(DrawElementsIndirectCommand_t)
        );
        COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // map batchIDs of batches that passed occlusion test to the `occlusionCulledMdiBuffer`
        driver->bindDescriptorSets(video::EPBP_COMPUTE, cullShaderData.mapPipeline->getLayout(), 0u, 1u, &cullShaderData.mapDS.get(), nullptr);
        driver->bindComputePipeline(cullShaderData.mapPipeline.get());
        
        driver->dispatchIndirect(cullShaderData.dispatchIndirect.get(), 0u);
        COpenGLExtensionHandler::pGlMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        // second fill visibility buffer pass
        driver->setRenderTarget(visBuffer);
        fillVBuffer(sceneData.occlusionCulledMdiBuffer);
        COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        //draw aabbs
        //draw3DLine->draw(camera->getConcatenatedMatrix(), dbgLines);

        // shade visibility buffer
        driver->bindDescriptorSets(video::EPBP_COMPUTE,sceneData.shadeVBufferPpln->getLayout(),0u,4u,vBuffDs,nullptr);
        driver->bindComputePipeline(sceneData.shadeVBufferPpln.get());
        {
            auto camPos = camera->getAbsolutePosition();

            driver->pushConstants(sceneData.shadeVBufferPpln->getLayout(),IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(core::vector3df),&camPos.X);
        }
        driver->dispatch((params.WindowSize.Width-1u)/SHADING_WG_SIZE_X+1u,(params.WindowSize.Height-1u)/SHADING_WG_SIZE_Y+1u,1u);
        COpenGLExtensionHandler::extGlMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT); // is GL_FRAMEBUFFER_BARRIER_BIT needed?

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