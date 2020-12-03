// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include <random>

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

#include "nbl/irrpack.h"
struct Vertex
{
    uint32_t boneID;
    float pos[3];
    uint8_t color[4];
    uint8_t uv[2];
    float normal[3];
} PACK_STRUCT;
#include "nbl/irrunpack.h"

#include <nbl/asset/CCPUMeshPacker.h>
#include "common.glsl"

template<typename T>
inline T getRandomNumber(T rangeBegin, T rangeEnd)
{
    assert(rangeBegin <= rangeEnd);

    static std::random_device rd;
    static std::mt19937 mt(rd());
    std::uniform_int_distribution<uint32_t> dist(rangeBegin, rangeEnd);

    return dist(mt);
}

IFrameBuffer* createDepthOnlyFrameBuffer(video::IVideoDriver* driver)
{
    core::smart_refctd_ptr<IGPUImageView> gpuImageViewDepthBuffer;
    {
        IGPUImage::SCreationParams imgInfo;
        imgInfo.format = EF_D16_UNORM;
        imgInfo.type = IGPUImage::ET_2D;
        imgInfo.extent.width = 64u;
        imgInfo.extent.height = 64u;
        imgInfo.extent.depth = 1u;
        imgInfo.mipLevels = 1u;
        imgInfo.arrayLayers = 1u;
        imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
        imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

        auto image = driver->createGPUImageOnDedMem(std::move(imgInfo), driver->getDeviceLocalGPUMemoryReqs());

        IGPUImageView::SCreationParams imgViewInfo;
        imgViewInfo.image = std::move(image);
        imgViewInfo.format = EF_D16_UNORM;
        imgViewInfo.viewType = IGPUImageView::ET_2D;
        imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
        imgViewInfo.subresourceRange.baseArrayLayer = 0u;
        imgViewInfo.subresourceRange.baseMipLevel = 0u;
        imgViewInfo.subresourceRange.layerCount = 1u;
        imgViewInfo.subresourceRange.levelCount = 1u;

        gpuImageViewDepthBuffer = driver->createGPUImageView(std::move(imgViewInfo));
    }

    auto frameBuffer = driver->addFrameBuffer();
    frameBuffer->attach(video::EFAP_DEPTH_ATTACHMENT, std::move(gpuImageViewDepthBuffer));

    return frameBuffer;
}

constexpr uint32_t TEST_CASE_COUNT = 5u;
constexpr uint32_t TEST_CASE_SUBGROUPS = 4u;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
#ifdef _NBL_DEBUG
    params.WindowSize = dimension2d<uint32_t>(1280, 720);
#else
    params.WindowSize = dimension2d<uint32_t>(64, 64);
#endif
    params.Fullscreen = false;
    params.Vsync = false;
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    auto device = createDeviceEx(params);

    if (!device)
        return 1; // could not create selected driver.

    QToQuitEventReceiver receiver;
    device->setEventReceiver(&receiver);

    auto* am = device->getAssetManager();
    video::IVideoDriver* driver = device->getVideoDriver();

#ifndef _NBL_DEBUG
    auto* depthFBO = createDepthOnlyFrameBuffer(driver);
#endif

    IAssetLoader::SAssetLoadParams lp;
#ifdef _NBL_DEBUG
    auto vertexShaderBundle_1 = am->getAsset("../test_1.vert", lp);
    auto vertexShaderBundle_2 = am->getAsset("../test_2.vert", lp);
    auto vertexShaderBundle_3 = am->getAsset("../test_3.vert", lp);
    auto vertexShaderBundle_4 = am->getAsset("../test_4.vert", lp);
    auto vertexShaderBundle_5 = am->getAsset("../test_5.vert", lp);
#else
    auto vertexShaderBundle_1 = am->getAsset("../benchmark_1.vert", lp);
    auto vertexShaderBundle_2 = am->getAsset("../benchmark_2.vert", lp);
    auto vertexShaderBundle_3 = am->getAsset("../benchmark_3.vert", lp);
    auto vertexShaderBundle_4 = am->getAsset("../benchmark_4.vert", lp);
    auto vertexShaderBundle_5 = am->getAsset("../benchmark_5.vert", lp);
#endif
    auto fragShaderBundle = am->getAsset("../dirLight.frag", lp);
    ICPUSpecializedShader* shaders[TEST_CASE_COUNT][2];
    shaders[0][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_1.getContents().begin()->get());
    shaders[0][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[1][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_2.getContents().begin()->get());
    shaders[1][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[2][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_3.getContents().begin()->get());
    shaders[2][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[3][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_4.getContents().begin()->get());
    shaders[3][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[4][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_5.getContents().begin()->get());
    shaders[4][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());

    core::vector<uint16_t> boneMatMaxCnt;

    auto createMeshBufferFromGeometryCreatorReturnData = [&](asset::IGeometryCreator::return_type& geometryObject, core::smart_refctd_ptr<ICPUMeshBuffer>& meshBuffer, uint32_t mbID ,uint32_t bonesCreatedCnt)
    {
        for (int i = 0; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
            meshBuffer->setVertexBufferBinding(std::move(geometryObject.bindings[i]), i);
        meshBuffer->setBoundingBox(geometryObject.bbox);
        meshBuffer->setIndexCount(geometryObject.indexCount);
        meshBuffer->setIndexType(geometryObject.indexType);
        meshBuffer->setIndexBufferBinding(std::move(geometryObject.indexBuffer));
        
        auto newInputParams = geometryObject.inputParams;
        newInputParams.enabledAttribFlags |= 0b10000u;
        newInputParams.enabledBindingFlags |= 0b10u;
        newInputParams.bindings[1].inputRate = EVIR_PER_VERTEX;
        newInputParams.bindings[1].stride = 0u;
        newInputParams.attributes[4].binding = 1u;
        newInputParams.attributes[4].format = EF_R32_UINT;
        newInputParams.attributes[4].relativeOffset = 0u;

        SBufferBinding<ICPUBuffer> boneIDBuffer;
        boneIDBuffer.offset = 0u;
        boneIDBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(meshBuffer->calcVertexCount() * sizeof(uint32_t));

        uint32_t* buffPtr = static_cast<uint32_t*>(boneIDBuffer.buffer->getPointer());
        for (int i = 0; i < meshBuffer->calcVertexCount(); i++)
            buffPtr[i] = bonesCreatedCnt + getRandomNumber<uint32_t>(1u, boneMatMaxCnt[mbID]) - 1u;
        // don't want total random access to bones, sort roughly 
        std::sort(buffPtr,buffPtr+meshBuffer->calcVertexCount());

        meshBuffer->setVertexBufferBinding(std::move(boneIDBuffer), 1);

        auto pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(nullptr, nullptr, nullptr, newInputParams, SBlendParams(), SPrimitiveAssemblyParams(), SRasterizationParams());
        meshBuffer->setPipeline(std::move(pipeline));
    };
    const auto MaxBufferSize = driver->getMaxSSBOSize();
#ifdef _NBL_DEBUG
    const core::vector4du32_SIMD diskBlockDim(5u, 5u, 5u);
#else
    const uint32_t multiplier = core::min<uint32_t>(MAX_OBJ_CNT*(MaxBufferSize>>20u)/(0x1u<<10u),MAX_OBJ_CNT);
    const core::vector4du32_SIMD diskBlockDim(1u, 1u, multiplier);
#endif
    const size_t diskCount = diskBlockDim.x * diskBlockDim.y * diskBlockDim.z;
    os::Printer::print("Disks count "+std::to_string(diskCount)+"\n");

    assert(diskCount <= MAX_OBJ_CNT);

    std::vector<uint16_t> tesselation(diskCount);
    
#ifdef _NBL_DEBUG
    constexpr uint32_t maxTesselation = 32u;
#else
    constexpr uint32_t maxTesselation = 16000u;
#endif
    {
        //get random tesselation for disks
        std::generate(tesselation.begin(), tesselation.end(), [&]() { return getRandomNumber<uint32_t>(15u, maxTesselation - 1u) | 0x0001; });

        //get random bone cnt for disks
        boneMatMaxCnt = core::vector<uint16_t>(diskCount);
        std::generate(boneMatMaxCnt.begin(), boneMatMaxCnt.end(), [&]() { return getRandomNumber<uint16_t>(1u, MAX_BONE_CNT);  });
    }

    const uint32_t boneMatrixCnt = std::accumulate(boneMatMaxCnt.begin(), boneMatMaxCnt.end(), 0u);

    assert(boneMatrixCnt < MAT_MAX_CNT);

    core::vector<uint16_t> indices(16000);
    std::iota(indices.begin(), indices.end(), 0u);

    core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> disks(diskCount);
    std::generate(disks.begin(), disks.end(), []() { return core::make_smart_refctd_ptr<ICPUMeshBuffer>(); });

#ifdef _NBL_DEBUG
    for (uint32_t i = 0u, bonesCreated = 0u; i < diskCount; i++)
    {
        auto disk = am->getGeometryCreator()->createDiskMesh(0.5f, tesselation[i]);
        auto newIdxBuffer = am->getMeshManipulator()->idxBufferFromTrianglesFanToTriangles(indices.data(), disk.indexCount, EIT_16BIT);
        disk.indexBuffer = { 0ull, newIdxBuffer };
        disk.indexCount = newIdxBuffer->getSize() / sizeof(uint16_t);
        disk.indexType = EIT_16BIT;
        disk.assemblyParams.primitiveType = EPT_TRIANGLE_LIST;
        createMeshBufferFromGeometryCreatorReturnData(disk, disks[i], i, bonesCreated);
        bonesCreated += boneMatMaxCnt[i];
    }
#else
    {
        auto disk = am->getGeometryCreator()->createDiskMesh(0.5f, maxTesselation);
        //reset input params, `createMeshBufferFromGeometryCreatorReturnData` will create vertex buffer only with boneID attribute
        disk.inputParams = SVertexInputParams();
        for (uint32_t i = 0u, bonesCreated = 0u; i < diskCount; i++)
        {
            auto newIdxBuffer = am->getMeshManipulator()->idxBufferFromTrianglesFanToTriangles(indices.data(), tesselation[i] + 2u, EIT_16BIT);
            disk.indexBuffer = { 0ull, newIdxBuffer };
            disk.indexCount = newIdxBuffer->getSize() / sizeof(uint16_t);
            disk.indexType = EIT_16BIT;
            disk.assemblyParams.primitiveType = EPT_TRIANGLE_LIST;
            createMeshBufferFromGeometryCreatorReturnData(disk, disks[i], i, bonesCreated);
            bonesCreated += boneMatMaxCnt[i];
            if (i%50u==1u)
                os::Printer::print("Disks progress " + std::to_string(float(i)/float(diskCount)*100.f) + "\%\n");
        }
    }
#endif


    os::Printer::print("Disks creation done.\n");

        //pack disks
    MeshPackerBase::PackedMeshBuffer<ICPUBuffer> packedMeshBuffer;
    MeshPackerBase::PackedMeshBufferData mb;
    {
        auto allocParams = MeshPackerBase::AllocationParams();
#ifdef _NBL_DEBUG
        allocParams.MDIDataBuffSupportedCnt = 1024;
        allocParams.MDIDataBuffMinAllocSize = 512;
        allocParams.indexBuffSupportedCnt = 8192 * 2;
        allocParams.indexBufferMinAllocSize = 256;
        allocParams.vertexBuffSupportedCnt = 8192;
        allocParams.vertexBufferMinAllocSize = 256;
#else
        allocParams.MDIDataBuffSupportedCnt = 500000 * 2;
        allocParams.MDIDataBuffMinAllocSize = 1024;
        allocParams.indexBuffSupportedCnt = MaxBufferSize/sizeof(uint16_t);
        allocParams.indexBufferMinAllocSize = 512 * 1024;
        allocParams.vertexBuffSupportedCnt = MaxBufferSize/sizeof(uint32_t);
        allocParams.vertexBufferMinAllocSize = 512 * 1024;
#endif

        CCPUMeshPacker packer(disks[0]->getPipeline()->getVertexInputParams(), allocParams, 256, 256);

        auto resParams = packer.alloc(disks.begin(), disks.end());

        _NBL_DEBUG_BREAK_IF(resParams.isValid() == false);

        packer.instantiateDataStorage();

        mb = packer.commit(disks.begin(), disks.end(), resParams);

        packedMeshBuffer = packer.getPackedMeshBuffer();

        _NBL_DEBUG_BREAK_IF(mb.isValid() == false);
        _NBL_DEBUG_BREAK_IF(packedMeshBuffer.isValid() == false);
    }

    disks.clear();
    disks.shrink_to_fit();

    os::Printer::print("Packing done.\n");

    struct DrawIndexedIndirectInput
    {
        asset::SBufferBinding<IGPUBuffer> vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        asset::E_PRIMITIVE_TOPOLOGY mode = EPT_TRIANGLE_LIST;
        asset::E_INDEX_TYPE indexType = EIT_16BIT; 
        core::smart_refctd_ptr<IGPUBuffer> indexBuff = nullptr;
        core::smart_refctd_ptr<IGPUBuffer> indirectDrawBuff = nullptr;
        size_t offset = 0u; 
        size_t maxCount = 0u;
        size_t stride = 0u;
        size_t countOffset = 0u;
    };

    //create inputs for `drawIndexedIndirect`
    DrawIndexedIndirectInput mdi;
    {
        mdi.indexBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packedMeshBuffer.indexBuffer.buffer->getSize(), packedMeshBuffer.indexBuffer.buffer->getPointer());

        _NBL_DEBUG_BREAK_IF(mb.mdiParameterCount == 0u);
        mdi.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(DrawElementsIndirectCommand_t)* mb.mdiParameterCount, packedMeshBuffer.MDIDataBuffer->getPointer());

        auto& cpuVtxBuff = packedMeshBuffer.vertexBufferBindings[4].buffer;
        auto gpuVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());

        for (uint32_t i = 0u; i < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
            mdi.vtxBindings[i] = { packedMeshBuffer.vertexBufferBindings[i].offset, gpuVtxBuff };

        mdi.maxCount = mb.mdiParameterCount;
    }

        //create bone matrices
    struct BoneNormalMatPair
    {
        core::matrix4SIMD boneMatrix;
        core::matrix3x4SIMD normalMatrix;
    };
    core::smart_refctd_ptr<IGPUBuffer> drawDataBuffer[TEST_CASE_COUNT];
    vector<core::matrix3x4SIMD> translationMatrices_2(diskCount);
    core::vector<core::matrix4SIMD> boneMatrices(boneMatrixCnt);
    core::vector<core::matrix3x4SIMD> normalMatrices(boneMatrixCnt);
    core::vector<BoneNormalMatPair> boneAndNormalMatrices(boneMatrixCnt);
    core::vector<core::vectorSIMDf> boneMatricesRows(boneMatrixCnt * 4);
    core::vector<core::vectorSIMDf> normalMatricesRows(boneMatrixCnt * 3);
    core::vector<float> boneMatricesComponents(boneMatrixCnt * 4 * 4);
    core::vector<float> normalMatricesComponents(boneMatrixCnt * 3 * 3);
    {
        uint32_t i = 0u;
        for (uint32_t x = 0; x < diskBlockDim.x; x++)
        for (uint32_t y = 0; y < diskBlockDim.y; y++)
        for (uint32_t z = 0; z < diskBlockDim.z; z++)
        {
            translationMatrices_2[i].setTranslation(core::vectorSIMDf(5.0f) * core::vectorSIMDf(x , y , z ));
            i++;
        }  

        //as packed matrices
        drawDataBuffer[0] = driver->createDeviceLocalGPUBufferOnDedMem(boneAndNormalMatrices.size() * sizeof(BoneNormalMatPair));

        //as separated matrices
        drawDataBuffer[1] = driver->createDeviceLocalGPUBufferOnDedMem(MAT_MAX_CNT * sizeof(core::matrix4SIMD) + MAT_MAX_CNT * sizeof(core::matrix3x4SIMD));

        //as vectors
        drawDataBuffer[2] = driver->createDeviceLocalGPUBufferOnDedMem((BONE_VEC_MAX_CNT + NORM_VEC_MAX_CNT) * sizeof(core::vectorSIMDf));

        //as floats
        drawDataBuffer[3] = driver->createDeviceLocalGPUBufferOnDedMem((BONE_COMP_MAX_CNT + NORM_COMP_MAX_CNT) * sizeof(float));

        drawDataBuffer[TEST_CASE_SUBGROUPS] = drawDataBuffer[0];
    }

    

    os::Printer::print("GPU memory allocation done.\n");
    
        //create pipeline
    struct Shader3PushConstants
    {
        core::vector4du32_SIMD matrixOffsets;
    };

    struct Shader4PushConstants
    {
        uint32_t matrixOffsets[16];
    };

    //TODO
    core::smart_refctd_ptr<IGPUPipelineLayout> gpuPipelineLayout[TEST_CASE_COUNT];
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline[TEST_CASE_COUNT];
    core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet[TEST_CASE_COUNT];

    Shader3PushConstants s3pc;
    s3pc.matrixOffsets = core::vector4du32_SIMD(0u, boneMatrixCnt, boneMatrixCnt * 2, boneMatrixCnt * 3);

    Shader4PushConstants s4pc;
    for (uint32_t i = 0u; i < 16; i++)
        s4pc.matrixOffsets[i] = i * boneMatrixCnt;

    {
        asset::SPushConstantRange range[TEST_CASE_COUNT] = {
            asset::ISpecializedShader::ESS_UNKNOWN, 0u, 0u,
            asset::ISpecializedShader::ESS_UNKNOWN, 0u, 0u,
            asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader3PushConstants),
            asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader4PushConstants),
            asset::ISpecializedShader::ESS_UNKNOWN, 0u, 0u
        };

        //TODO
        for (uint32_t i = 0u; i < TEST_CASE_COUNT; i++)
        {
            core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout;
            {
                video::IGPUDescriptorSetLayout::SBinding b[2];
                b[0].binding = 0u;
                b[0].count = 1u;
                b[0].type = EDT_STORAGE_BUFFER;
                b[1] = b[0];
                b[1].binding = 1u;

                uint32_t count = i == TEST_CASE_SUBGROUPS ? 2u : 1u;
                layout = driver->createGPUDescriptorSetLayout(b, b + count);
            }

            descriptorSet[i] = driver->createGPUDescriptorSet(core::smart_refctd_ptr(layout));
            {
                video::IGPUDescriptorSet::SWriteDescriptorSet w[2];
                w[0].binding = 0u;
                w[0].arrayElement = 0u;
                w[0].count = 1u;
                w[0].descriptorType = EDT_STORAGE_BUFFER;
                w[0].dstSet = descriptorSet[i].get();
                w[1] = w[0];

                video::IGPUDescriptorSet::SDescriptorInfo info;
                info.buffer.offset = 0u;
                info.buffer.size = drawDataBuffer[i]->getSize();
                info.desc = drawDataBuffer[i];

                w[0].info = &info;
                w[1].info = &info;

                uint32_t count = i == TEST_CASE_SUBGROUPS ? 2u : 1u;
                driver->updateDescriptorSets(count, w, 0u, nullptr);
            }
            
            auto gpuShaders = driver->getGPUObjectsFromAssets(shaders[i], shaders[i] + 2);
            IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

            if(i == 0u || i == 1u)
                gpuPipelineLayout[i] = driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(layout));
            else
                gpuPipelineLayout[i] = driver->createGPUPipelineLayout(&range[i], &range[i] + 1, core::smart_refctd_ptr(layout));

            asset::SRasterizationParams rasterParams;
            rasterParams.faceCullingMode = asset::EFCM_NONE;
#ifndef _NBL_DEBUG
            rasterParams.faceCullingMode = asset::EFCM_BACK_BIT;
            rasterParams.depthTestEnable = true;
            rasterParams.depthWriteEnable = false;
#endif

            SBlendParams blendParams;
#ifndef _NBL_DEBUG
            blendParams.blendParams[0].colorWriteMask = 0u;
#endif

#ifndef _NBL_DEBUG
            constexpr uint32_t shaderCnt = 1u;
#else
            constexpr uint32_t shaderCnt = 2u;
#endif

            gpuPipeline[i] = driver->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(gpuPipelineLayout[i]), shaders, shaders + shaderCnt, packedMeshBuffer.vertexInputParams, blendParams, SPrimitiveAssemblyParams(), rasterParams);
        }
    }

    auto smgr = device->getSceneManager();

    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(0.01f);
    camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);

    device->getCursorControl()->setVisible(false);

    uint64_t lastFPSTime = 0;
    float lastFastestMeshFrameNr = -1.f;

    core::vector<float> rps(diskCount);
    std::generate(rps.begin(), rps.end(), [&]() { return 8.0f * 3.14f / (static_cast<float>(getRandomNumber<uint32_t>(1u, 20u)) - 10.1f); });
    
    size_t timeMS = 0ull;

    auto constructMatrices = [&]()
    {
        for (uint32_t i = 0u, currMatOffset = 0u; i < translationMatrices_2.size(); i++)
        {
            core::matrix3x4SIMD modelMatrix;
            modelMatrix.setRotation(core::quaternion(0.0f, rps[i] * (timeMS / 1000.0f), 0.0f));
            modelMatrix.concatenateBefore(translationMatrices_2[i]);

            boneMatrices[currMatOffset] = core::concatenateBFollowedByA(camera->getConcatenatedMatrix(), modelMatrix);

            normalMatrices[currMatOffset] = modelMatrix;

            for (uint32_t j = 1u; j < boneMatMaxCnt[i]; j++)
            {
                boneMatrices[currMatOffset + j] = boneMatrices[currMatOffset];
                normalMatrices[currMatOffset + j] = normalMatrices[currMatOffset];
            }

            currMatOffset += boneMatMaxCnt[i];
        }

        for (uint32_t i = 0u; i < boneAndNormalMatrices.size(); i++)
        {
            boneAndNormalMatrices[i].boneMatrix = boneMatrices[i];
            boneAndNormalMatrices[i].normalMatrix = core::matrix3x4SIMD(normalMatrices[i].pointer());
        }

        for (uint32_t i = 0u; i < boneMatrixCnt; i++)
        {
            boneMatricesRows[s3pc.matrixOffsets[0] + i] = boneMatrices[i].getRow(0);
            boneMatricesRows[s3pc.matrixOffsets[1] + i] = boneMatrices[i].getRow(1);
            boneMatricesRows[s3pc.matrixOffsets[2] + i] = boneMatrices[i].getRow(2);
            boneMatricesRows[s3pc.matrixOffsets[3] + i] = boneMatrices[i].getRow(3);

            normalMatricesRows[s3pc.matrixOffsets[0] + i] = core::matrix4SIMD(normalMatrices[i]).getRow(0);
            normalMatricesRows[s3pc.matrixOffsets[1] + i] = core::matrix4SIMD(normalMatrices[i]).getRow(1);
            normalMatricesRows[s3pc.matrixOffsets[2] + i] = core::matrix4SIMD(normalMatrices[i]).getRow(2);
        }

        for (uint32_t i = 0u; i < boneMatrixCnt; i++)
        {
            for (uint32_t j = 0u; j < 16u; j++)
                boneMatricesComponents[s4pc.matrixOffsets[j] + i] = *(boneMatrices[i].pointer() + j);

            for (uint32_t j = 0u; j < 9u; j++)
                normalMatricesComponents[s4pc.matrixOffsets[j] + i] = core::matrix4SIMD(normalMatrices[i]).getRow(j / 3u)[(j + 3u) % 3];
        }
    };

    auto updatePushConstants = [&](uint32_t caseID)
    {
        switch (caseID)
        {
        case 0: [[fallthrough]];
        case 1: [[fallthrough]];
        case TEST_CASE_SUBGROUPS:
        break;
        case 2:
            driver->pushConstants(gpuPipelineLayout[2].get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader3PushConstants), &s3pc);
        break;
        case 3:
            driver->pushConstants(gpuPipelineLayout[3].get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader4PushConstants), &s4pc);
        break;
        default:
            assert(false);
        }
    };
    
    auto updateSSBO = [&](uint32_t caseID)
    {
        switch (caseID)
        {
        case 0: [[fallthrough]];
        case TEST_CASE_SUBGROUPS:
        {
            const size_t matricesByteSize = sizeof(BoneNormalMatPair) * boneAndNormalMatrices.size();

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[0].get(), 0u, matricesByteSize, boneAndNormalMatrices.data());
        }
        break;
        case 1:
        {
            const size_t boneMatricesByteSize = sizeof(core::matrix4SIMD) * boneMatrices.size();
            const size_t normalMatricesByteSize = sizeof(core::matrix3x4SIMD) * normalMatrices.size();
            const size_t normalMatOffset = sizeof(core::matrix4SIMD) * MAT_MAX_CNT;

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[1].get(), 0u, boneMatricesByteSize, boneMatrices.data());
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[1].get(), normalMatOffset, normalMatricesByteSize, normalMatrices.data());
        }
        break;
        case 2:
        {
            const size_t boneMatricesRowsByteSize = sizeof(core::vectorSIMDf) * boneMatricesRows.size();
            const size_t normalMatricesRowsByteSize = sizeof(core::vectorSIMDf) * normalMatricesRows.size();
            const size_t normalMatricesOffset = sizeof(core::vectorSIMDf) * BONE_VEC_MAX_CNT;

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[2].get(), 0u, boneMatricesRowsByteSize, boneMatricesRows.data());
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[2].get(), normalMatricesOffset, normalMatricesRowsByteSize, normalMatricesRows.data());
        }
        break;
        case 3:
        {
            const size_t boneMatricesCompByteSize = sizeof(float) * boneMatricesComponents.size();
            const size_t normalMatricesCompByteSize = sizeof(float) * normalMatricesComponents.size();
            const size_t normalMatricesOffset = sizeof(float) * BONE_COMP_MAX_CNT;

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[3].get(), 0u, boneMatricesCompByteSize, boneMatricesComponents.data());
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[3].get(), normalMatricesOffset, normalMatricesCompByteSize, normalMatricesComponents.data());
        }
        break;
        default:
            assert(false);
        }
    };

    std::function<bool()> exitCondition;
#ifdef _NBL_DEBUG
    exitCondition = [&]() { return device->run() && receiver.keepOpen(); };
#else
    exitCondition = []() { return true; };
#endif

    COpenGLDriver* driverOGL = dynamic_cast<COpenGLDriver*>(driver);

    constexpr uint32_t iterationCnt = 1000u;
    constexpr uint32_t warmupIterationCnt = iterationCnt / 10u;
    for (uint32_t caseID = 0u; caseID < TEST_CASE_COUNT; caseID++)
    {
        os::Printer::print(std::string("Benchmark for case nr. " + std::to_string(caseID)));

        driver->bindGraphicsPipeline(gpuPipeline[caseID].get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline[caseID]->getLayout(), 0u, 1u, &descriptorSet[caseID].get(), nullptr);
        updatePushConstants(caseID);

#ifndef _NBL_DEBUG
        driver->beginScene(true, true, video::SColor(0, 0, 0, 255));
        driver->setRenderTarget(depthFBO);
        driver->clearZBuffer(1.0f);

        for (uint32_t i = 0u; i < warmupIterationCnt; i++)
            driver->drawIndexedIndirect(mdi.vtxBindings, mdi.mode, mdi.indexType, mdi.indexBuff.get(), mdi.indirectDrawBuff.get(), mdi.offset, mdi.maxCount, mdi.stride);
#endif
        float avg = 0.0f;
        for (uint32_t j = 0u; j < 5u && exitCondition(); j++)
        {
            IQueryObject* query = driver->createElapsedTimeQuery();
            driver->beginQuery(query);
            for (uint32_t i = 0u; i < iterationCnt && exitCondition(); i++)
            {
#ifdef _NBL_DEBUG
                driver->beginScene(true, true, video::SColor(0, 0, 0, 255));
                timeMS = std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count();

                camera->OnAnimate(timeMS);
                camera->render();

                constructMatrices();

                updateSSBO(caseID);
#endif
                driver->drawIndexedIndirect(mdi.vtxBindings, mdi.mode, mdi.indexType, mdi.indexBuff.get(), mdi.indirectDrawBuff.get(), mdi.offset, mdi.maxCount, mdi.stride);
#ifdef _NBL_DEBUG
                driver->endScene();
#endif
            }
            glFlush();
            driver->endQuery(query);
            driver->setRenderTarget(nullptr);

            uint32_t timeElapsed = 0u;
            query->getQueryResult(&timeElapsed);
            query->drop();

            os::Printer::print(std::string("Result ") + std::to_string(j) + std::string(": ") + std::to_string(static_cast<double>(timeElapsed) / 1000000.0) + std::string("ms."));
            avg += static_cast<double>(timeElapsed) / 1000000.0;

            if (j == 4)
                os::Printer::print(std::string("Avg time: ") + std::to_string(avg / 5.0f) + std::string("\n"));
        }

    }
#ifndef _NBL_DEBUG
    os::Printer::print(std::string("Type Something to Exit:"));
    std::cin.get();
#endif
    return 0;
}
