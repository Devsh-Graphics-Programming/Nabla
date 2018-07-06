#include <numeric>
#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "../source/Irrlicht/COpenGLPersistentlyMappedBuffer.h"
#include "createComputeShader.h"

// benchmark controls
#define TEST_CASE 3 // [1..4]
#define INCPUMEM false
#define DONT_UPDATE_BUFFER 0 // 0 or 1

using namespace irr;
using namespace core;

int main()
{
    srand(time(nullptr));

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

    GLuint shaders[16];
    for (size_t i = 0u; i < 16u; ++i)
        shaders[i] = createComputeShaderFromFile(("../cs" + std::to_string(i)).c_str());

#define CS_WG_SIZE 1024
    size_t wgBudget = 3000000 / CS_WG_SIZE;
    size_t wgCnts[100];
    for (size_t i = 0u; i < 100u; ++i)
    {
        const size_t r = rand() % 15 + 15;
        if (r > wgBudget)
            wgCnts[i] = wgBudget;
        else
            wgCnts[i] = r;
        wgBudget -= wgCnts[i];
    }
    if (wgBudget > 99u)
    {
        for (size_t i = 0u; i < 99u; ++i)
            wgCnts[i] += wgBudget / 100u;
    }
    for (size_t i = 0u; i < 100u; ++i)
        wgCnts[i] = std::ceil(std::sqrt((float)wgCnts[i]));

    size_t batchSizes[16];
    memset(batchSizes, 0, sizeof(batchSizes));
    {
        const size_t invocCntMax = 180000u;
        size_t b = 0u;

        size_t invocCnt = 0u;
        for (size_t i = 0u; i < 100u; ++i)
        {
            if (invocCnt + wgCnts[i]*CS_WG_SIZE <= invocCntMax || b == 15u)
            {
                ++batchSizes[b];
                invocCnt += wgCnts[i] * CS_WG_SIZE;
            }
            else
            {
                invocCnt = 0u;
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
            batchesPsum[i] = batchesPsum[i - 1] + batchSizes[i];

        size_t b = 0u;
        for (size_t i = 1u; i < 100u; ++i)
        {
            if (i >= batchesPsum[b])
                ++b;
            offsets[i] = offsets[i - 1] + wgCnts[i - 1] * uboStructSizes[b];
        }
    }

    const size_t bufSize = offsets[99] + wgCnts[99] * uboStructSizes[15];
#if (TEST_CASE==3 || TEST_CASE==4)
#if DONT_UPDATE_BUFFER
    const size_t persistentlyMappedBufSize = bufSize;
#else
    const size_t persistentlyMappedBufSize = 4 * bufSize;
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

    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(driver)->getThreadContext());
    size_t frameNum = 0u;

#define ITER_CNT 1000
    video::IQueryObject* queries[ITER_CNT];

#if ((TEST_CASE==3 || TEST_CASE==4) && !DONT_UPDATE_BUFFER)
    video::IDriverFence* fences[4]{ nullptr, nullptr, nullptr, nullptr };
#endif

    while (device->run() && frameNum < ITER_CNT)
    {
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
            auto waitf = [&frameNum, &fences] {
                auto res = fences[frameNum % 4]->waitCPU(10000000000ull);
                return (res == video::EDFR_CONDITION_SATISFIED || res == video::EDFR_ALREADY_SIGNALED);
            };
            while (!waitf())
            {
                fences[frameNum % 4]->drop();
                fences[frameNum % 4] = nullptr;
            }
        }
        memcpy(((uint8_t*)(dynamic_cast<video::COpenGLPersistentlyMappedBuffer*>(buffer)->getPointer())) + (frameNum % 4)*bufSize, cpubuffer->getPointer(), bufSize);
#if TEST_CASE==3
        video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(buffer->getOpenGLName(), (frameNum % 4)*bufSize, bufSize);
#endif //TEST_CASE==3
#endif //!DONT_UPDATE_BUFFER
#endif // this large #if/#elif/#elif

        size_t i = 0u;
        for (size_t j = 0u; j < 16u; ++j)
        {
            auto glbuf = static_cast<const video::COpenGLBuffer*>(buffer);
            video::COpenGLExtensionHandler::extGlUseProgram(shaders[j]);
            size_t tmp = i;
            for (; i < tmp + batchSizes[j]; ++i)
            {
                const ptrdiff_t sz = (i == 99u ? (ptrdiff_t)bufSize : offsets[i + 1]) - offsets[i];
                const ptrdiff_t off = offsets[i]
#if (TEST_CASE==3 || TEST_CASE==4)
                    + (frameNum % 4) * bufSize;
#else
                    ;
#endif
                auxCtx->setActiveUBO(0u, 1u, &glbuf, &off, &sz);
                video::COpenGLExtensionHandler::extGlDispatchCompute(wgCnts[i], wgCnts[i], 1);
            }
        }

#if ((TEST_CASE==3 || TEST_CASE==4) && !DONT_UPDATE_BUFFER)
        if (!fences[frameNum % 4])
            fences[frameNum % 4] = driver->placeFence();
#endif

        driver->endQuery(queries[frameNum]);

#if (TEST_CASE==1 && !DONT_UPDATE_BUFFER)
        buffer->drop();
        buffer = nullptr;
#endif

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

    for (size_t i = 0u; i < 16u; ++i)
        video::COpenGLExtensionHandler::extGlDeleteProgram(shaders[i]);
#if TEST_CASE!=1
    buffer->drop();
#endif
    cpubuffer->drop();

    device->drop();

    return 0;
}