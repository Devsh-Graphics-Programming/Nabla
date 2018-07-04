#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "../source/Irrlicht/COpenGLPersistentlyMappedBuffer.h"


#define TEST_CASE 0
#define INCPUMEM false

using namespace irr;
using namespace core;


int main()
{
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

    video::E_MATERIAL_TYPE material[16];
    for (size_t i = 0u; i < 16u; ++i)
    {
        material[i] = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles(
            ("../vs" + std::to_string(i)).c_str(),
            "", "", "",
            "../fs",
            3, video::EMT_SOLID);
    }

#define INSTANCE_CNT 10

    const size_t batches[16]{ 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 1u, 1u };
    const size_t sizes[16]{ 144u, 144u, 144u, 144u, 128u, 128u, 144u, 80u, 80u, 128u, 112u, 96u, 144u, 96u, 144u, 96u };
    ptrdiff_t offsets[100]{ 0u };
    {
        size_t i = 1u;
        for (size_t j = 0u; j < 16u; ++j)
        {
            size_t tmp = i;
            for (; i < tmp + (i < 98u ? 7u : 1u /* thats ugly af, i'm sorry*/); ++i)
                offsets[i] = offsets[i - 1] + sizes[j] * (i-1u < 98u ? 7u : 1u /* thats ugly af, i'm sorry*/) * INSTANCE_CNT;
        }
    }
    const size_t bufSize = [&sizes,&batches] {
        size_t s = 0u;
        for (size_t i = 0u; i < 16u; ++i)
            s += sizes[i] * batches[i] * INSTANCE_CNT;
        return s;
    }();

    core::ICPUBuffer* cpubuffer = new core::ICPUBuffer(bufSize);
    video::COpenGLBuffer* buffer;
#if TEST_CASE==0
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createGPUBuffer(bufSize, nullptr, true, INCPUMEM));
#elif TEST_CASE==3
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createPersistentlyMappedBuffer(bufSize*4, nullptr, video::EGBA_WRITE, false, INCPUMEM));
#elif TEST_CASE==4
    buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createPersistentlyMappedBuffer(bufSize*4, nullptr, video::EGBA_WRITE, true, INCPUMEM));
#endif

    scene::IGPUMeshBuffer* meshes[100];
    scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
    desc->mapVertexAttrBuffer(buffer, scene::EVAI_ATTR0, scene::ECPA_FOUR, scene::ECT_BYTE); // map whatever buffer just to activate whatever vertex attribute (look below)
    // however some other buffer needed for TEST_CASE==1... (todo)
    for (size_t i = 0u; i < 100u; ++i)
    {
        meshes[i] = new scene::IGPUMeshBuffer();
        meshes[i]->setInstanceCount(INSTANCE_CNT);
        meshes[i]->setIndexCount(3 * 1000000 / (INSTANCE_CNT * 100));
        meshes[i]->setMeshDataAndFormat(desc); // apparently glDrawArrays does nothing if no vertex attribs are active
    }
    desc->drop();

    uint64_t lastFPSTime = 0;

    video::SMaterial smaterial;
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(driver)->getThreadContext());
    while (device->run())
    {
        driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

        video::IQueryObject* query = driver->createElapsedTimeQuery();
        driver->beginQuery(query);

        memset(cpubuffer->getPointer(), 0, bufSize);
#if TEST_CASE==0
        buffer->updateSubRange(0u, bufSize, cpubuffer->getPointer());
#elif TEST_CASE==1
        buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createGPUBuffer(bufSize, cpubuffer->getPointer(), false, INCPUMEM));
#elif (TEST_CASE==3 || TEST_CASE==4)
        //FENCE
        memcpy(dynamic_cast<video::COpenGLPersistentlyMappedBuffer*>(buffer)->getPointer(), cpubuffer->getPointer(), bufSize);
        //FLUSH
#endif

        size_t i = 0u;
        for (size_t j = 0u; j < 16u; ++j)
        {
            auto glbuf = static_cast<const video::COpenGLBuffer*>(buffer);
            smaterial.MaterialType = material[j];
            driver->setMaterial(smaterial);
            size_t tmp = i;
            for (; i < tmp + batches[j]; ++i)
            {
                ptrdiff_t sz = sizes[j] * batches[j] * INSTANCE_CNT;
                auxCtx->setActiveUBO(0u, 1u, &glbuf, offsets+i, &sz);
                driver->drawMeshBuffer(meshes[i]);
            }
        }

#if TEST_CASE==1
        buffer->drop();
        buffer = nullptr;
#endif

        driver->endQuery(query);
        uint32_t dt = 0u;
        query->getQueryResult(&dt);
        os::Printer::log("delta time", std::to_string(dt).c_str());

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream sstr;
            sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();
            //wprintf(L"%s\n", sstr.str().c_str());
            device->setWindowCaption(sstr.str().c_str());
            lastFPSTime = time;
        }
    }

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
