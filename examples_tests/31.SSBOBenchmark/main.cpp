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

#include "common.glsl"
#include "commonIndirect.glsl"

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

    auto vertexShaderBundle = am->getAsset("../ssboBenchmarkShaders/vertexShader.vert", lp);
    auto fragmentShaderBundle = am->getAsset("../ssboBenchmarkShaders/fragmentShader.frag", lp);

    CShaderIntrospector introspector(am->getGLSLCompiler());
    const auto extensions = driver->getSupportedGLSLExtensions();

    //create cpu shaders
    ICPUSpecializedShader* shaders[2] =
    {
        IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().first->get()),
        IAsset::castDown<ICPUSpecializedShader>(fragmentShaderBundle.getContents().first->get())
    };
    auto cpuPipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders, shaders + 2, extensions->begin(), extensions->end());

    //temporary
    constexpr size_t diskCount = 3001;

    std::vector<uint16_t> tesselation(diskCount);

    //get random tesselation for disks
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<uint32_t> dist(15, 14999);

        //TODO: test
        //`dist(mt) | 0x0001` so vertexCount is always odd (only when diskCount is odd as well)
        std::generate(tesselation.begin(), tesselation.end(), [&]() { return dist(mt) | 0x0001; });
    }

    //set mesh params
    const size_t vertexCount = std::accumulate<core::vector<uint16_t>::iterator, size_t>(tesselation.begin(), tesselation.end(), 0) + 2 * diskCount; //sum + 2 * diskCount because vertex count of each disk is equal to: (tesselation + 2)
    const size_t indexCount  = std::accumulate<core::vector<uint16_t>::iterator, size_t>(tesselation.begin(), tesselation.end(), 0) * 3;
    E_INDEX_TYPE indexType = E_INDEX_TYPE::EIT_32BIT;
    constexpr uint32_t diskVertexSize = 30;
    constexpr uint32_t vertexSize = sizeof(Vertex);
    SBufferBinding<ICPUBuffer> vertexBuffer{ 0, core::make_smart_refctd_ptr<ICPUBuffer>(vertexCount * vertexSize) };
    // SBufferBinding<ICPUBuffer> bindings[ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
    SBufferBinding<ICPUBuffer> indexBuffer{ 0, core::make_smart_refctd_ptr<ICPUBuffer>(indexCount * sizeof(uint32_t)) };

    SPrimitiveAssemblyParams assemblyParams;
    assemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;

    SVertexInputParams vertexInputParams = 
    { 
        0b1111u, 0b1u,
        {
            {0u,EF_R32_UINT,offsetof(Vertex,boneID)},
            {0u,EF_R32G32B32_SFLOAT,offsetof(Vertex,pos)},
            {0u,EF_R8G8B8A8_UNORM,offsetof(Vertex,color)},
            {0u,EF_R8G8_USCALED,offsetof(Vertex,uv)},
            {0u,EF_R32G32B32_SFLOAT,offsetof(Vertex,normal)}
        },
        {vertexSize,EVIR_PER_VERTEX} 
    };


    core::vector<core::matrix3x4SIMD> SSBOData; //bone matrices will also reside here
    SSBOData.reserve(diskCount);

    //create disks and join them into one mesh buffer
    {
        uint8_t* vtxBuffPtr = static_cast<uint8_t*>(vertexBuffer.buffer.get()->getPointer());
        uint32_t* idxBuffPtr = static_cast<uint32_t*>(indexBuffer.buffer.get()->getPointer());

        size_t objectID = 0;
        for (uint16_t tess : tesselation)
        {
            auto disk = am->getGeometryCreator()->createDiskMesh(1.5f, tess);
            uint8_t* oldVertexPtr = static_cast<uint8_t*>(disk.bindings[0].buffer.get()->getPointer());

            //TODO: test
            //fill vertex buffer
            for (uint16_t i = 0; i < tess; i++)
            {
                *reinterpret_cast<uint32_t*>(vtxBuffPtr) = objectID;
                vtxBuffPtr += sizeof(uint32_t);

                memcpy(vtxBuffPtr, oldVertexPtr, diskVertexSize);
                vtxBuffPtr += diskVertexSize;
            }

            //TODO: test
            //fill index buffer
            for (int i = 0, nextVertex = 1; i < 3 * tess; i += 3)
            {
                idxBuffPtr[0] = 0;
                idxBuffPtr[1] = nextVertex++;
                idxBuffPtr[2] = (nextVertex > tess) ? 1 : nextVertex;

                idxBuffPtr += 3;
            }

            objectID++;
        }

        //set model matrices
        for (int x = 0; x < (diskCount - 1) / 3; x++)
        for (int y = 0; y < (diskCount - 1) / 3; y++)
        for (int z = 0; z < (diskCount - 1) / 3; z++)
            SSBOData.emplace_back(core::matrix3x4SIMD().setTranslation(core::vectorSIMDf(x * 2, y * 2, z * 2)));

        SSBOData.emplace_back(core::matrix3x4SIMD().setTranslation(core::vectorSIMDf(diskCount * 2)));

        //
    }

        //create shader pipeline
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipeline;
    //gpuPipeline = driver->getGPUObjectsFromAssets(&cpuDiskDrawDirectPipeline.get(), &cpuDiskDrawDirectPipeline.get() + 1)->operator[](0);

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
        //if (device->isWindowActive())
    {
        driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        core::matrix3x4SIMD normalMatrix;
        camera->getViewMatrix().getSub3x3InverseTranspose(normalMatrix);

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream str;
            str << L"SSBO Benchmark - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(str.str());
            lastFPSTime = time;
        }
    }

    return 0;
}
