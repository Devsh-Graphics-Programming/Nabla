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

#include "C:\IrrlichtBAW\IrrlichtBAW\src\irr\asset\CCPUMeshPacker.h";

//#include "common.glsl"
//#include "commonIndirect.glsl"

        //TODO: create shader, and then from check how introspector creates renderpass


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


    auto createMeshBufferFromGeometryCreatorReturnData = [&](asset::IGeometryCreator::return_type& geometryObject, ICPUMeshBuffer* meshBuffer)
    {
        asset::SBlendParams blendParams;
        asset::SRasterizationParams rasterParams;
        rasterParams.faceCullingMode = asset::EFCM_NONE;

        auto pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(nullptr, nullptr, nullptr, geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

        meshBuffer->setPipeline(std::move(pipeline));
        for (int i = 0; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
            meshBuffer->setVertexBufferBinding(std::move(geometryObject.bindings[i]), i);
        meshBuffer->setBoundingBox(geometryObject.bbox);
        meshBuffer->setIndexCount(geometryObject.indexCount);
        meshBuffer->setIndexType(geometryObject.indexType);
        meshBuffer->setIndexBufferBinding(std::move(geometryObject.indexBuffer));
    };

    auto createMeshBufferFromMeshPackerOutput = [&](MeshPackerBase::PackedMeshBuffer<ICPUBuffer>& packedMeshBuffer, ICPUMeshBuffer* meshBuffer)
    {

        asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
        auto pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range, range + 1);

        asset::SBlendParams blendParams;
        asset::SRasterizationParams rasterParams;
        rasterParams.faceCullingMode = asset::EFCM_NONE;

        auto pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(pipelineLayout), shaders, shaders + 2, packedMeshBuffer.vertexInputParams, blendParams, SPrimitiveAssemblyParams(), rasterParams);

        meshBuffer->setPipeline(std::move(pipeline));
        for (int i = 0; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
            meshBuffer->setVertexBufferBinding(std::move(packedMeshBuffer.vertexBufferBindings[i]), i);
        //meshBuffer->setBoundingBox(packedMeshBuffer.MDIDataBuffer.bbox);

        DrawElementsIndirectCommand_t firstMDIStruct = *static_cast<DrawElementsIndirectCommand_t*>(packedMeshBuffer.MDIDataBuffer->getPointer());

        meshBuffer->setIndexCount(firstMDIStruct.count);
        meshBuffer->setIndexType(EIT_16BIT);
        meshBuffer->setIndexBufferBinding(std::move(packedMeshBuffer.indexBuffer));
    };

        //create disks
    ICPUMeshBuffer* meshBuffer[2] = { _IRR_NEW(ICPUMeshBuffer), _IRR_NEW(ICPUMeshBuffer) };
    auto sphere1 = am->getGeometryCreator()->createSphereMesh();
    auto sphere2 = am->getGeometryCreator()->createSphereMesh(1.0f);

    createMeshBufferFromGeometryCreatorReturnData(sphere1, meshBuffer[0]);
    createMeshBufferFromGeometryCreatorReturnData(sphere2, meshBuffer[1]);

        //pack disks into single buffers
    MeshPackerBase::PackedMeshBuffer<ICPUBuffer> packedMeshBuffer;
    MeshPackerBase::PackedMeshBufferData mb;
    {
        auto allocParams = MeshPackerBase::AllocationParams();
        allocParams.MDIDataBuffSupportedCnt = 16;
        allocParams.MDIDataBuffMinAllocSize = 1;
        allocParams.indexBuffSupportedCnt = 2048 * 2;
        allocParams.indexBufferMinAllocSize = 128;
        allocParams.vertexBuffSupportedCnt = 2048 * 2;
        allocParams.vertexBufferMinAllocSize = 128;

        CCPUMeshPacker packer(meshBuffer[0]->getPipeline()->getVertexInputParams(), allocParams, 128, 128);

        auto resParams = packer.alloc(meshBuffer, meshBuffer + 2);

        _IRR_DEBUG_BREAK_IF(resParams.isValid() == false);

        packer.instantiateDataStorage();
        mb = packer.commit(meshBuffer, meshBuffer + 2, resParams);

        packedMeshBuffer = packer.getPackedMeshBuffer();

        _IRR_DEBUG_BREAK_IF(mb.isValid() == false);
        _IRR_DEBUG_BREAK_IF(packedMeshBuffer.isValid() == false);
    }

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

    

        //create `drawIndirect` data
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
    {
        vector<core::matrix4SIMD> matrices(2);
        matrices[1].setTranslation(core::vectorSIMDf(0.0f, 5.5f, 0.0f));

        drawDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(matrices.size() * sizeof(core::matrix4SIMD), matrices.data());
    }
    
        //create pipeline
    core::smart_refctd_ptr<IGPUPipelineLayout> gpuPipelineLayout;
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    {
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout;
        {
            video::IGPUDescriptorSetLayout::SBinding b[1];
            b[0].binding = 0u;
            b[0].count = 1u;
            b[0].type = EDT_STORAGE_BUFFER;
            
            layout = driver->createGPUDescriptorSetLayout(b, b + 1);
        }

        asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
        gpuPipelineLayout = driver->createGPUPipelineLayout(range, range + 1, core::smart_refctd_ptr(layout));

        core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet = driver->createGPUDescriptorSet(std::move(layout));
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

    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(0, 0, 0, 255));

        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setRotation(core::quaternion(0.0f, (90.0f * 2.0f * 3.14f) / 360.0f, 0.0f));
        modelMatrix.setTranslation(irr::core::vectorSIMDf(0, 0, 0, 0));

        core::matrix4SIMD vp = camera->getConcatenatedMatrix();
        driver->bindGraphicsPipeline(gpuPipeline.get());
        driver->pushConstants(gpuPipelineLayout.get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), vp.pointer());
        driver->drawIndexedIndirect(mdi.vtxBindings, mdi.mode, mdi.indexType, mdi.indexBuff.get(), mdi.indirectDrawBuff.get(), mdi.offset, mdi.maxCount, mdi.stride);

        driver->endScene();
    }

    return 0;
}
