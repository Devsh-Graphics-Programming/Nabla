#define _IRR_STATIC_LIB_
#include <iostream>
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

//#include "../source/Irrlicht/COpenGLTextureBufferObject.h"

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
    auto vertexShaderBundle = am->getAsset("../basic.vert", lp);
    auto fragShaderBundle = am->getAsset("../basic.frag", lp);
    ICPUSpecializedShader* shaders[2] = { IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get()),IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get()) };
    auto gpuShaders = driver->getGPUObjectsFromAssets(shaders, shaders + 2);


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

    const core::vector4du32_SIMD diskBlockDim(5u, 5u, 5u);
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
    core::smart_refctd_ptr<IGPUBuffer> drawDataBuffer;
    vector<core::matrix4SIMD> translationMatrices(diskCount + 1);
    {

        uint32_t i = 1u;
        for (uint32_t x = 0; x < diskBlockDim.x; x++)
        for (uint32_t y = 0; y < diskBlockDim.y; y++)
        for (uint32_t z = 0; z < diskBlockDim.z; z++)
        {
            translationMatrices[i].setTranslation(core::vectorSIMDf(5.0f) * core::vectorSIMDf(x , y , z ));
            i++;
        }  

        drawDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(translationMatrices.size() * sizeof(core::matrix4SIMD), translationMatrices.data());
    }

    os::Printer::print("GPU memory allocation done.\n");
    
        //create pipeline
    core::smart_refctd_ptr<IGPUPipelineLayout> gpuPipelineLayout;
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
    {
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout;
        {
            video::IGPUDescriptorSetLayout::SBinding b[1];
            b[0].binding = 0u;
            b[0].count = 1u;
            b[0].type = EDT_STORAGE_BUFFER;
            
            layout = driver->createGPUDescriptorSetLayout(b, b + 1);
        }
        
        descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(layout));
        {
            video::IGPUDescriptorSet::SWriteDescriptorSet w;
            w.binding = 0u;
            w.arrayElement = 0u;
            w.count = 1u;
            w.descriptorType = EDT_STORAGE_BUFFER;
            w.dstSet = descriptorSet.get();

            video::IGPUDescriptorSet::SDescriptorInfo info;
            info.buffer.offset = 0u;
            info.buffer.size = drawDataBuffer->getSize();
            info.desc = drawDataBuffer;

            w.info = &info;

            driver->updateDescriptorSets(1u, &w, 0u, nullptr);
        }
        
        asset::SRasterizationParams rasterParams;
        rasterParams.faceCullingMode = asset::EFCM_NONE;
        IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() }; //fix?

        asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
        gpuPipelineLayout = driver->createGPUPipelineLayout(range, range + 1, core::smart_refctd_ptr(layout));

        gpuPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), shaders, shaders + 2, packedMeshBuffer.vertexInputParams, SBlendParams(), SPrimitiveAssemblyParams(), rasterParams);
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

        // ----------------------------- MAIN LOOP ----------------------------- 

    core::vector<core::matrix4SIMD> boneMatrices(diskCount + 1);
    core::vector<float> rps(diskCount);
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<uint32_t> dist(1, 20);

        std::generate(rps.begin(), rps.end(), [&]() { return 8.0f * 3.14f / (static_cast<float>(dist(mt)) - 10.1f); });
    }

    size_t timeMS = 0ull;
    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(0, 0, 0, 255));

        timeMS = std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count();
        camera->OnAnimate(timeMS);
        camera->render();

        boneMatrices[0] = camera->getConcatenatedMatrix();

        for (uint32_t i = 1; i < translationMatrices.size(); i++)
        {
            core::matrix3x4SIMD rotationMatrix;
            rotationMatrix.setRotation(core::quaternion(0.0f, rps[i - 1] * (timeMS / 1000.0f), 0.0f));
            boneMatrices[i] = core::concatenateBFollowedByA(translationMatrices[i], core::matrix4SIMD(rotationMatrix));
        }
            

        driver->bindGraphicsPipeline(gpuPipeline.get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 0u, 1u, &descriptorSet.get(), nullptr);
        driver->updateBufferRangeViaStagingBuffer(drawDataBuffer.get(), 0u, sizeof(core::matrix4SIMD) * boneMatrices.size(), boneMatrices.data());
        //driver->pushConstants(gpuPipelineLayout.get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        driver->drawIndexedIndirect(mdi.vtxBindings, mdi.mode, mdi.indexType, mdi.indexBuff.get(), mdi.indirectDrawBuff.get(), mdi.offset, mdi.maxCount, mdi.stride);

        driver->endScene();
    }

    return 0;
}
