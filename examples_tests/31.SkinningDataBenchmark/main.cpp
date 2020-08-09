#define _IRR_STATIC_LIB_
#include <iostream>
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include <random>

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

#include "irr/irrpack.h"
struct Vertex
{
    uint32_t boneID;
    float pos[3];
    uint8_t color[4];
    uint8_t uv[2];
    float normal[3];
} PACK_STRUCT;
#include "irr/irrunpack.h"

#include "C:\IrrlichtBAW\IrrlichtBAW\src\irr\asset\CCPUMeshPacker.h"; //sorry

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

    IAssetLoader::SAssetLoadParams lp;
    auto vertexShaderBundle_1 = am->getAsset("../1.vert", lp);
    auto vertexShaderBundle_2 = am->getAsset("../2.vert", lp);
    auto vertexShaderBundle_3 = am->getAsset("../3.vert", lp);
    auto vertexShaderBundle_4 = am->getAsset("../4.vert", lp);

    auto fragShaderBundle = am->getAsset("../dirLight.frag", lp);
    ICPUSpecializedShader* shaders[4][2];
    shaders[0][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_1.getContents().begin()->get());
    shaders[0][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[1][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_2.getContents().begin()->get());
    shaders[1][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[2][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_3.getContents().begin()->get());
    shaders[2][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());
    shaders[3][0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle_4.getContents().begin()->get());
    shaders[3][1] = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());

    auto createMeshBufferFromGeometryCreatorReturnData = [&](asset::IGeometryCreator::return_type& geometryObject, ICPUMeshBuffer* meshBuffer, uint16_t boneID)
    {
        asset::SBlendParams blendParams;
        asset::SRasterizationParams rasterParams;
        rasterParams.faceCullingMode = asset::EFCM_NONE;

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
        newInputParams.attributes[4].format = EF_R16_UINT;
        newInputParams.attributes[4].relativeOffset = 0u;

        SBufferBinding<ICPUBuffer> boneIDBuffer;
        boneIDBuffer.offset = 0u;
        boneIDBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(meshBuffer->calcVertexCount() * sizeof(uint16_t));

        uint16_t* buffPtr = static_cast<uint16_t*>(boneIDBuffer.buffer->getPointer());
        for (int i = 0; i < meshBuffer->calcVertexCount(); i++)
            *(buffPtr + i) = boneID;


        meshBuffer->setVertexBufferBinding(std::move(boneIDBuffer), 1);

        auto pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(nullptr, nullptr, nullptr, newInputParams, blendParams, geometryObject.assemblyParams, rasterParams);
        meshBuffer->setPipeline(std::move(pipeline));
    };

    const core::vector4du32_SIMD diskBlockDim(3u, 3u, 3u);
    const size_t diskCount = diskBlockDim.x * diskBlockDim.y * diskBlockDim.z;

    std::vector<uint16_t> tesselation(diskCount);

    //get random tesselation for disks
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<uint32_t> dist(15, 160 - 1);

        //TODO: test
        //`dist(mt) | 0x0001` so vertexCount is always odd (only when diskCount is odd as well)
        std::generate(tesselation.begin(), tesselation.end(), [&]() { return dist(mt) | 0x0001; });
    }

    core::vector<uint16_t> indices(16000);
    std::iota(indices.begin(), indices.end(), 0u);

    core::vector<ICPUMeshBuffer*> disks(diskCount);
    std::generate(disks.begin(), disks.end(), []() { return _IRR_NEW(ICPUMeshBuffer); });
    for (uint32_t i = 0; i < diskCount; i++)
    {
        auto disk = am->getGeometryCreator()->createDiskMesh(0.5f, tesselation[i]);
        auto newIdxBuffer = am->getMeshManipulator()->idxBufferFromTrianglesFanToTriangles(indices.data(), disk.indexCount, EIT_16BIT);
        disk.indexBuffer = { 0ull, newIdxBuffer };
        disk.indexCount = newIdxBuffer->getSize() / sizeof(uint16_t);
        disk.indexType = EIT_16BIT;
        disk.assemblyParams.primitiveType = EPT_TRIANGLE_LIST;
        createMeshBufferFromGeometryCreatorReturnData(disk, disks[i], i);
    }

    os::Printer::print("Disks creation done.\n");

        //pack disks
    MeshPackerBase::PackedMeshBuffer<ICPUBuffer> packedMeshBuffer;
    MeshPackerBase::PackedMeshBufferData mb;
    {
        auto allocParams = MeshPackerBase::AllocationParams();
        allocParams.MDIDataBuffSupportedCnt = 1024;
        allocParams.MDIDataBuffMinAllocSize = 512;
        allocParams.indexBuffSupportedCnt = 1000 * 160 * 10;
        allocParams.indexBufferMinAllocSize = 256;
        allocParams.vertexBuffSupportedCnt = 1000 * 160 * 10;
        allocParams.vertexBufferMinAllocSize = 256;

        CCPUMeshPacker packer(disks[0]->getPipeline()->getVertexInputParams(), allocParams, 256, 256);

        auto resParams = packer.alloc(disks.begin(), disks.end());

        _IRR_DEBUG_BREAK_IF(resParams.isValid() == false);

        packer.instantiateDataStorage();
        mb = packer.commit(disks.begin(), disks.end(), resParams);

        packedMeshBuffer = packer.getPackedMeshBuffer();

        _IRR_DEBUG_BREAK_IF(mb.isValid() == false);
        _IRR_DEBUG_BREAK_IF(packedMeshBuffer.isValid() == false);
    }

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
        core::smart_refctd_ptr<IGPUBuffer> countBuffer = nullptr;
        size_t countOffset = 0u;
    };

        //create inputs for `drawIndexedIndirect`
    DrawIndexedIndirectInput mdi;
    {
        mdi.indexBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packedMeshBuffer.indexBuffer.buffer->getSize(), packedMeshBuffer.indexBuffer.buffer->getPointer());

        _IRR_DEBUG_BREAK_IF(mb.mdiParameterCount == 0u);
        mdi.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(DrawElementsIndirectCommand_t)* mb.mdiParameterCount, packedMeshBuffer.MDIDataBuffer->getPointer());

        auto& cpuVtxBuff = packedMeshBuffer.vertexBufferBindings[0].buffer;
        auto gpuVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());

        for (uint32_t i = 0u; i < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
            mdi.vtxBindings[i] = { packedMeshBuffer.vertexBufferBindings[i].offset, gpuVtxBuff };

        mdi.maxCount = mb.mdiParameterCount;
    }

        //create bone matrices
    core::smart_refctd_ptr<IGPUBuffer> drawDataBuffer[4];
    vector<core::matrix3x4SIMD> translationMatrices_2(diskCount);
    core::vector<core::matrix4SIMD> boneMatrices(diskCount);
    core::vector<core::matrix4SIMD> normalMatrices(diskCount);
    core::vector<core::matrix4SIMD> boneAndNormalMatrices(diskCount * 2);
    core::vector<core::vectorSIMDf> boneMatricesRows(diskCount * 4);
    core::vector<core::vectorSIMDf> normalMatricesRows(diskCount * 3);
    core::vector<float> boneMatricesComponents(diskCount * 4 * 4);
    core::vector<float> normalMatricesComponents(diskCount * 3 * 3);
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
        drawDataBuffer[0] = driver->createDeviceLocalGPUBufferOnDedMem(boneAndNormalMatrices.size() * sizeof(core::matrix4SIMD));

        //as separated matrices
        drawDataBuffer[1] = driver->createDeviceLocalGPUBufferOnDedMem(boneMatrices.size() * sizeof(core::matrix4SIMD) * 2u);

        //as vectors
        drawDataBuffer[2] = driver->createDeviceLocalGPUBufferOnDedMem((boneMatricesRows.size() + normalMatricesRows.size()) * sizeof(core::vectorSIMDf));

        //as floats
        drawDataBuffer[3] = driver->createDeviceLocalGPUBufferOnDedMem((boneMatricesComponents.size() + normalMatricesComponents.size()) * sizeof(float));
    }

    

    os::Printer::print("GPU memory allocation done.\n");
    
        //create pipeline
    struct Shader1PushConstants
    {
        core::matrix4SIMD vp;
    };

    struct Shader2PushConstants
    {
        uint32_t normalMatrixArrayOffset;
    };

    struct Shader3PushConstants
    {
        core::vector4du32_SIMD matrixOffsets;
        uint32_t normalMatrixArrayOffset;
    };

    struct Shader4PushConstants
    {
        uint32_t matrixOffsets[16];
    };

    core::smart_refctd_ptr<IGPUPipelineLayout> gpuPipelineLayout[4];
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline[4];
    core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet[4];
    
    Shader1PushConstants s1pc;

    Shader2PushConstants s2pc;
    s2pc.normalMatrixArrayOffset = boneMatrices.size() * sizeof(core::matrix4SIMD);

    Shader3PushConstants s3pc;
    s3pc.matrixOffsets = core::vector4du32_SIMD(0u, diskCount, diskCount * 2, diskCount * 3);
    s3pc.normalMatrixArrayOffset = boneMatricesRows.size();

    Shader4PushConstants s4pc;
    for (uint32_t i = 0u; i < 16; i++)
        s4pc.matrixOffsets[i] = i * diskCount;

    {
        asset::SPushConstantRange range[4] = {
            asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader1PushConstants),
            asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader2PushConstants), 
            asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader3PushConstants),
            asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader4PushConstants)
        };

        for (uint32_t i = 0u; i < 4u; i++)
        {
            core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout;
            {
                video::IGPUDescriptorSetLayout::SBinding b[1];
                b[0].binding = 0u;
                b[0].count = 1u;
                b[0].type = EDT_STORAGE_BUFFER;

                layout = driver->createGPUDescriptorSetLayout(b, b + 1);
            }

            descriptorSet[i] = driver->createGPUDescriptorSet(core::smart_refctd_ptr(layout));
            {
                video::IGPUDescriptorSet::SWriteDescriptorSet w;
                w.binding = 0u;
                w.arrayElement = 0u;
                w.count = 1u;
                w.descriptorType = EDT_STORAGE_BUFFER;
                w.dstSet = descriptorSet[i].get();

                video::IGPUDescriptorSet::SDescriptorInfo info;
                info.buffer.offset = 0u;
                info.buffer.size = drawDataBuffer[i]->getSize();
                info.desc = drawDataBuffer[i];

                w.info = &info;

                driver->updateDescriptorSets(1u, &w, 0u, nullptr);
            }

            asset::SRasterizationParams rasterParams;
            rasterParams.faceCullingMode = asset::EFCM_NONE;
            
            auto gpuShaders = driver->getGPUObjectsFromAssets(shaders[i], shaders[i] + 2);
            IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

            gpuPipelineLayout[i] = driver->createGPUPipelineLayout(&range[i], &range[i] + 1, core::smart_refctd_ptr(layout));

            gpuPipeline[i] = driver->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(gpuPipelineLayout[i]), shaders, shaders + 2, packedMeshBuffer.vertexInputParams, SBlendParams(), SPrimitiveAssemblyParams(), rasterParams);
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
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<uint32_t> dist(1, 20);

        std::generate(rps.begin(), rps.end(), [&]() { return 8.0f * 3.14f / (static_cast<float>(dist(mt)) - 10.1f); });
    }
    
    size_t timeMS = 0ull;

    enum ShaderToUse
    {
        packedMatrices,
        separateMatrices,
        rows,
        components
    };

    constexpr ShaderToUse shaderToUse = components;

    switch (shaderToUse)
    {
    case packedMatrices:
        driver->bindGraphicsPipeline(gpuPipeline[0].get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline[0]->getLayout(), 0u, 1u, &descriptorSet[0].get(), nullptr);
        break;
    case separateMatrices:
        driver->bindGraphicsPipeline(gpuPipeline[1].get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline[1]->getLayout(), 0u, 1u, &descriptorSet[1].get(), nullptr);
        break;
    case rows:
        driver->bindGraphicsPipeline(gpuPipeline[2].get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline[2]->getLayout(), 0u, 1u, &descriptorSet[2].get(), nullptr);
        break;
    case components:
        driver->bindGraphicsPipeline(gpuPipeline[3].get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline[3]->getLayout(), 0u, 1u, &descriptorSet[3].get(), nullptr);
        break;
    default:
        assert(false);
    }
    
    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(0, 0, 0, 255));

        timeMS = std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count();
        camera->OnAnimate(timeMS);
        camera->render();

        //construct model matrices
        for (uint32_t i = 0u, j = 0u; i < translationMatrices_2.size(); i++)
        {
            core::matrix3x4SIMD modelMatrix;
            modelMatrix.setRotation(core::quaternion(0.0f, rps[i] * (timeMS / 1000.0f), 0.0f));
            modelMatrix.concatenateBefore(translationMatrices_2[i]);
            
            boneMatrices[i] = core::concatenateBFollowedByA(camera->getConcatenatedMatrix(), modelMatrix);
            normalMatrices[i] = core::matrix4SIMD(modelMatrix);

            boneAndNormalMatrices[j++] = boneMatrices[i];
            boneAndNormalMatrices[j++] = normalMatrices[i];
        }

        //update ssbo
        switch (shaderToUse)
        {
        case packedMatrices:
        {
            const size_t matricesByteSize = sizeof(core::matrix4SIMD) * boneAndNormalMatrices.size();

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[0].get(), 0u, matricesByteSize, boneAndNormalMatrices.data());

            s1pc.vp = camera->getConcatenatedMatrix();
            driver->pushConstants(gpuPipelineLayout[0].get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader1PushConstants), &s1pc);
        }
        break;
        case separateMatrices:
        {
            const size_t boneMatricesByteSize = sizeof(core::matrix4SIMD) * boneMatrices.size();

            //update bone matrices
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[1].get(), 0u, boneMatricesByteSize, boneMatrices.data());
            //update normal matrices (normal matrices are almost identical to bone matrices so I will store bone matrices there, translation part of bone matrices will be ignored anyway)
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[1].get(), boneMatricesByteSize, boneMatricesByteSize, normalMatrices.data());

            driver->pushConstants(gpuPipelineLayout[1].get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader2PushConstants), &s2pc);
        }
        break;
        case rows:
        {
            //create array of separated matrix rows
            for (uint32_t i = 0u; i < diskCount; i++)
            {
                boneMatricesRows[s3pc.matrixOffsets[0] + i] = boneMatrices[i].getRow(0);
                boneMatricesRows[s3pc.matrixOffsets[1] + i] = boneMatrices[i].getRow(1);
                boneMatricesRows[s3pc.matrixOffsets[2] + i] = boneMatrices[i].getRow(2);
                boneMatricesRows[s3pc.matrixOffsets[3] + i] = boneMatrices[i].getRow(3);

                normalMatricesRows[s3pc.matrixOffsets[0] + i] = normalMatrices[i].getRow(0);
                normalMatricesRows[s3pc.matrixOffsets[1] + i] = normalMatrices[i].getRow(1);
                normalMatricesRows[s3pc.matrixOffsets[2] + i] = normalMatrices[i].getRow(2);
            }

            const size_t boneMatricesRowsByteSize = sizeof(core::vectorSIMDf) * boneMatricesRows.size();
            const size_t normalMatricesRowsByteSize = sizeof(core::vectorSIMDf) * normalMatricesRows.size();

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[2].get(), 0u, boneMatricesRowsByteSize, boneMatricesRows.data());
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[2].get(), boneMatricesRowsByteSize, normalMatricesRowsByteSize, normalMatricesRows.data());
            driver->pushConstants(gpuPipelineLayout[2].get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader3PushConstants), &s3pc);
        }
        break;
        case components:
        {
            for (uint32_t i = 0u; i < diskCount; i++)
            {
                for (uint32_t j = 0u; j < 16u; j++)
                    boneMatricesComponents[s4pc.matrixOffsets[j] + i] = *(boneMatrices[i].pointer() + j);

                for (uint32_t j = 0u; j < 9u; j++)
                    normalMatricesComponents[s4pc.matrixOffsets[j] + i] = *(boneMatrices[i].pointer() + j); //fix!
            }

            const size_t boneMatricesCompByteSize = sizeof(float) * boneMatricesComponents.size();
            const size_t normalMatricesCompByteSize = sizeof(float) * normalMatricesComponents.size();

            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[3].get(), 0u, boneMatricesCompByteSize, boneMatricesComponents.data());
            driver->updateBufferRangeViaStagingBuffer(drawDataBuffer[3].get(), boneMatricesCompByteSize, normalMatricesCompByteSize, normalMatricesComponents.data());
            driver->pushConstants(gpuPipelineLayout[3].get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(Shader4PushConstants), &s4pc);
        }
        break;
        default:
            assert(false);
        }

        driver->drawIndexedIndirect(mdi.vtxBindings, mdi.mode, mdi.indexType, mdi.indexBuff.get(), mdi.indirectDrawBuff.get(), mdi.offset, mdi.maxCount, mdi.stride);
        
        driver->endScene();
    }

    return 0;
}
