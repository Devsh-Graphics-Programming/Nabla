#include <numeric>
#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "../source/Irrlicht/COpenGLPersistentlyMappedBuffer.h"

// benchmark controls
#define TEST_CASE 3 // [1..4]
#define INCPUMEM false
#define RASTERIZER_DISCARD false
#define DONT_UPDATE_BUFFER 0 // 0 or 1

using namespace irr;
using namespace core;


int main()
{
    srand(time(nullptr));

    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    irr::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(64, 64);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    IrrlichtDevice* device = createDeviceEx(params);

    if (!device)
        return 1; 

    video::IVideoDriver* driver = device->getVideoDriver();
    uint32_t depthBufSz[]{ 64u, 64u };
    video::ITexture* depthBuffer = driver->addTexture(video::ITexture::ETT_2D, depthBufSz, 1, "Depth", video::ECF_DEPTH32F);

    video::E_MATERIAL_TYPE material[16];
    for (size_t i = 0u; i < 16u; ++i)
    {
        material[i] = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles(
            ("../vs" + std::to_string(i)).c_str(),
            "", "", "",
            "../fs",
            3, video::EMT_SOLID);
    }

    scene::IGPUMeshBuffer* meshes[100];
    scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
    auto attrBuf = driver->createGPUBuffer(4 * 30000, nullptr);
    desc->mapVertexAttrBuffer(attrBuf, scene::EVAI_ATTR0, scene::ECPA_ONE, scene::ECT_FLOAT); // map whatever buffer just to activate whatever vertex attribute (look below)
    {
        size_t triBudget = 1600000u; //1.6M
        for (size_t i = 0u; i < 100u; ++i)
        {
            meshes[i] = new scene::IGPUMeshBuffer();
            const size_t instCnt = rand() % 10 + 1;
            const size_t triCnt = rand() % 3000 + 3000/3;
            meshes[i]->setInstanceCount(instCnt);
            if (instCnt * triCnt > triBudget)
                meshes[i]->setIndexCount(std::max((size_t)30, triBudget/instCnt*3));
            else
                meshes[i]->setIndexCount(3 * triCnt);
            triBudget -= (meshes[i]->getInstanceCount() * meshes[i]->getIndexCount())/3;

            meshes[i]->setMeshDataAndFormat(desc); // apparently glDrawArrays does nothing if no vertex attribs are active
        }
    }
    desc->drop();

    size_t batchSizes[16];
    memset(batchSizes, 0, sizeof(batchSizes));
    {
        const size_t triCntMax = 100000u; //100k
        size_t b = 0u;

        size_t triCnt = 0u;
        for (size_t i = 0u; i < 100u; ++i)
        {
            if (triCnt + (meshes[i]->getInstanceCount() * meshes[i]->getIndexCount())/3 <= triCntMax || b == 15u)
            {
                ++batchSizes[b];
                triCnt += (meshes[i]->getInstanceCount() * meshes[i]->getIndexCount())/3;
            }
            else
            {
                triCnt = 0u;
                ++b;
            }
        }
        const size_t sum = std::accumulate(batchSizes, batchSizes + 16, 0ull);
        for (size_t i = 0; i < 100u - sum; ++i)
            batchSizes[15 - (i % 16)]++;
    }

    const size_t uboStructSizes[16]{ 144u, 144u, 144u, 144u, 128u, 128u, 144u, 80u, 80u, 128u, 112u, 96u, 144u, 96u, 144u, 96u };
    ptrdiff_t offsets[100]{ 0 };
    {
        size_t batchesPsum[16]{ batchSizes[0] };
        for (size_t i = 1u; i < 16u; ++i)
            batchesPsum[i] = batchesPsum[i-1] + batchSizes[i];

        size_t b = 0u;
        for (size_t i = 1u; i < 100u; ++i)
        {
            if (i >= batchesPsum[b])
                ++b;
            offsets[i] = offsets[i-1] + meshes[i-1]->getInstanceCount() * uboStructSizes[b];
        }
    }
    const size_t bufSize = offsets[99] + meshes[99]->getInstanceCount() * uboStructSizes[15];
#if (TEST_CASE==3 || TEST_CASE==4)
#if DONT_UPDATE_BUFFER
    const size_t persistentlyMappedBufSize = bufSize;
#else
    const size_t persistentlyMappedBufSize = 4*bufSize;
#endif
#endif

    core::ICPUBuffer* cpubuffer = new core::ICPUBuffer(bufSize);
    for (size_t i = 0u; i < bufSize / 2; ++i)
        ((uint16_t*)(cpubuffer->getPointer()))[i] = rand();

    video::COpenGLBuffer* buffer = nullptr;
#if ((TEST_CASE==1 || TEST_CASE==2) && DONT_UPDATE_BUFFER)
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createGPUBuffer(bufSize, cpubuffer->getPointer(), false, INCPUMEM));
#elif TEST_CASE==2
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createGPUBuffer(bufSize, cpubuffer->getPointer(), true, INCPUMEM));
#elif TEST_CASE==3
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createPersistentlyMappedBuffer(persistentlyMappedBufSize, nullptr, video::EGBA_WRITE, false, INCPUMEM));
#elif TEST_CASE==4
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createPersistentlyMappedBuffer(persistentlyMappedBufSize, nullptr, video::EGBA_WRITE, true, INCPUMEM));
#endif

#if ((TEST_CASE==3 || TEST_CASE==4) && DONT_UPDATE_BUFFER)
    memcpy(dynamic_cast<video::COpenGLPersistentlyMappedBuffer*>(buffer)->getPointer(), cpubuffer->getPointer(), persistentlyMappedBufSize);
#if TEST_CASE==3
    video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(buffer->getOpenGLName(), 0, bufSize);
#endif
#endif

    //GLint meshVao[100];

    video::IFrameBuffer* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_DEPTH_ATTACHMENT, depthBuffer);

    video::SMaterial smaterial;
    smaterial.RasterizerDiscard = RASTERIZER_DISCARD;
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(driver)->getThreadContext());
    size_t frameNum = 0u;

#define ITER_CNT 1000
    video::IQueryObject* queries[ITER_CNT];

#if ((TEST_CASE==3 || TEST_CASE==4) && !DONT_UPDATE_BUFFER)
    video::IDriverFence* fences[4]{ nullptr, nullptr, nullptr, nullptr };
#endif

    while (device->run() && frameNum < ITER_CNT)
    {
        driver->beginScene(false, false);
        driver->setRenderTarget(fbo, true);

        queries[frameNum] = driver->createElapsedTimeQuery();
        driver->beginQuery(queries[frameNum]);

#if (TEST_CASE==1 && !DONT_UPDATE_BUFFER)
        buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createGPUBuffer(bufSize, cpubuffer->getPointer(), false, INCPUMEM));
#elif TEST_CASE==2
#if !DONT_UPDATE_BUFFER
        buffer->updateSubRange(0u, bufSize, cpubuffer->getPointer());
#endif //!DONT_UPDATE_BUFFER
#elif (TEST_CASE==3 || TEST_CASE==4)
#if !DONT_UPDATE_BUFFER
        if (fences[frameNum % 4])
        {
            auto res = fences[frameNum % 4]->waitCPU(10000000000ull);
            if (res == video::EDFR_CONDITION_SATISFIED || res == video::EDFR_ALREADY_SIGNALED)
            {
                fences[frameNum % 4]->drop();
                fences[frameNum % 4] = nullptr;
            }
        }
        memcpy(((uint8_t*)(dynamic_cast<video::COpenGLPersistentlyMappedBuffer*>(buffer)->getPointer()))+(frameNum%4)*bufSize, cpubuffer->getPointer(), bufSize);
#if TEST_CASE==3
        video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(buffer->getOpenGLName(), (frameNum%4)*bufSize, bufSize);
#endif //TEST_CASE==3
#endif //!DONT_UPDATE_BUFFER
        {
        auto glbuf = static_cast<const video::COpenGLBuffer*>(buffer);
        const ptrdiff_t sz = bufSize;
#if DONT_UPDATE_BUFFER
        const ptrdiff_t off = 0;
#else
        const ptrdiff_t off = (frameNum%4)*bufSize;
#endif //DONT_UPDATE_BUFFER
        auxCtx->setActiveUBO(0u, 1u, &glbuf, &off, &sz);
        }
#endif

        size_t i = 0u;
        for (size_t j = 0u; j < 16u; ++j)
        {
            auto glbuf = static_cast<const video::COpenGLBuffer*>(buffer);
            smaterial.MaterialType = material[j];
            driver->setMaterial(smaterial);
            size_t tmp = i;
            for (; i < tmp + batchSizes[j]; ++i)
            {
                const ptrdiff_t sz = (i == 99u ? (ptrdiff_t)bufSize : offsets[i+1]) - offsets[i];
                const ptrdiff_t off = offsets[i]
#if (TEST_CASE==3 || TEST_CASE==4)
                    + (frameNum % 4) * bufSize;
#else
                    ;
#endif
                auxCtx->setActiveUBO(0u, 1u, &glbuf, &off, &sz);
                driver->drawMeshBuffer(meshes[i]);
                //if (!frameNum)
                //    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, meshVao+i);
                //video::COpenGLExtensionHandler::extGlEnableVertexArrayAttrib(meshVao[i], 0u);
            }
        }

#if ((TEST_CASE==3 || TEST_CASE==4) && !DONT_UPDATE_BUFFER)
        if (!fences[frameNum%4])
            fences[frameNum%4] = driver->placeFence();
#endif

        driver->endQuery(queries[frameNum]);

#if (TEST_CASE==1 && !DONT_UPDATE_BUFFER)
        buffer->drop();
        buffer = nullptr;
#endif
        driver->endScene();

        ++frameNum;
    }

    size_t elapsed = 0u;
    for (size_t i = 0u; i < ITER_CNT; ++i)
    {
        uint32_t res{};
        queries[i]->getQueryResult(&res);
        elapsed += res;
    }
    os::Printer::log("Elapsed time", std::to_string(elapsed).c_str());

    for (size_t i = 0u; i < 100u; ++i)
        meshes[i]->drop();
#if TEST_CASE!=1
    buffer->drop();
#endif
    cpubuffer->drop();

    //device->sleep(3000);
    device->drop();

    return 0;
}
