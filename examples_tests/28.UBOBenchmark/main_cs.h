#include "createComputeShader.h"


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

#ifdef OPENGL_DEBUG
    if (video::COpenGLExtensionHandler::FeatureAvailable[video::COpenGLExtensionHandler::IRR_KHR_debug])
    {
        glEnable(GL_DEBUG_OUTPUT);
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        video::COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        video::COpenGLExtensionHandler::pGlDebugMessageCallback(openGLCBFunc,NULL);
    }
    else
    {
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        video::COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        video::COpenGLExtensionHandler::pGlDebugMessageCallbackARB(openGLCBFunc,NULL);
    }
#endif // OPENGL_DEBUG

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

    asset::ICPUBuffer* cpubuffer = new asset::ICPUBuffer(bufSize);
    for (size_t i = 0u; i < bufSize / 2; ++i)
        ((uint16_t*)(cpubuffer->getPointer()))[i] = rand();

    video::IGPUBuffer* buffer = nullptr;
    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    reqs.memoryHeapLocation = INCPUMEM ? video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL:video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
#if TEST_CASE==2
    reqs.vulkanReqs.size = bufSize;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
    #if DONT_UPDATE_BUFFER
        #error "Need a discardable staging buffer upload here."
    #else
        buffer = driver->createGPUBufferOnDedMem(reqs,true);
        buffer->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size), cpubuffer->getPointer());
    #endif // DONT_UPDATE_BUFFER
#elif TEST_CASE==3
    reqs.vulkanReqs.size = persistentlyMappedBufSize;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE;
    buffer = driver->createGPUBufferOnDedMem(reqs,false);
    buffer->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_WRITE,video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size));
#elif TEST_CASE==4
    reqs.vulkanReqs.size = persistentlyMappedBufSize;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE|video::IDriverMemoryAllocation::EMCF_COHERENT;
    buffer = driver->createGPUBufferOnDedMem(reqs,false);
    buffer->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_WRITE,video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size));
#endif

#if DONT_UPDATE_BUFFER
#if (TEST_CASE==3 || TEST_CASE==4)
    memcpy(buffer->getBoundMemory()->getMappedPointer(), cpubuffer->getPointer(), persistentlyMappedBufSize);
#if TEST_CASE==3
    video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(static_cast<video::COpenGLBuffer*>(buffer)->getOpenGLName(), 0, bufSize);
#endif
    buffer->getBoundMemory()->unmapMemory();
#endif
#endif
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT);

    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(driver)->getThreadContext());
    size_t frameNum = 0u;

#define ITER_CNT 50000
    video::IQueryObject* queries[ITER_CNT];

#if ((TEST_CASE==3 || TEST_CASE==4) && !DONT_UPDATE_BUFFER)
    video::IDriverFence* fences[4]{ nullptr, nullptr, nullptr, nullptr };
#endif

    auto cpustart = std::chrono::steady_clock::now();
    while (device->run() && frameNum < ITER_CNT)
    {
        queries[frameNum] = driver->createElapsedTimeQuery();
        driver->beginQuery(queries[frameNum]);

#if !DONT_UPDATE_BUFFER
#if TEST_CASE==2
        buffer->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0u, bufSize), cpubuffer->getPointer());
#elif (TEST_CASE==3 || TEST_CASE==4)
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
        memcpy(reinterpret_cast<uint8_t*>(buffer->getBoundMemory()->getMappedPointer())+(frameNum%4)*bufSize, cpubuffer->getPointer(), bufSize);
#if TEST_CASE==3
        video::COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(static_cast<video::COpenGLBuffer*>(buffer)->getOpenGLName(), (frameNum % 4)*bufSize, bufSize);
#endif //TEST_CASE==3
#endif // this large #if/#elif/#elif
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT);
#endif //!DONT_UPDATE_BUFFER

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

        glFlush();
        ++frameNum;
    }
    auto diff = std::chrono::steady_clock::now() - cpustart;
    std::cout << "Elapsed CPU time " << std::chrono::duration <double, std::milli> (diff).count() << " ms\n";

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
    buffer->drop();
    cpubuffer->drop();

    device->drop();

    return 0;
}
