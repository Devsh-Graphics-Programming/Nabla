
#include "nbl/video/COpenGL_Swapchain.h"

#include "nbl/video/surface/CSurfaceGL.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/COpenGLImage.h"
#include "nbl/video/COpenGL_Queue.h"

#include "nbl/system/IThreadHandler.h"

namespace nbl::video
{

template <typename FunctionTableType_, typename ThreadHandler>
core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType_, ThreadHandler>> COpenGL_Swapchain<FunctionTableType_, ThreadHandler>::create(SCreationParams&& params,
    core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev,
    const egl::CEGL* _egl,
    ImagesArrayType&& images,
    const COpenGLFeatureMap* _features,
    EGLContext _ctx,
    EGLConfig _config,
    COpenGLDebugCallback* _dbgCb)
{
    if (!images || !images->size())
        return nullptr;
    if (images->size() < params.minImageCount)
        return nullptr;
    if (images->size() > MaxImages)
        return nullptr;

    auto extent = asset::VkExtent3D{ params.width, params.height };
    for (auto& img : (*images))
    {
        auto& ci = img->getCreationParameters();
        if (ci.type != asset::IImage::ET_2D)
            return nullptr;
        if (ci.arrayLayers != params.arrayLayers)
            return nullptr;
        if (ci.mipLevels != 1u)
            return nullptr;
        if (ci.extent.width != extent.width)
            return nullptr;
        if (ci.extent.height != extent.height)
            return nullptr;
    }

    auto* sc = new COpenGL_Swapchain<FunctionTableType>(std::move(params), std::move(dev), _egl, std::move(images), _features, _ctx, _config, _dbgCb);
    return core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>>(sc, core::dont_grab);
}

template <typename FunctionTableType_, typename ThreadHandler>
core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType_, ThreadHandler>> COpenGL_Swapchain<FunctionTableType_, ThreadHandler>::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{
    if (params.surface->getAPIType() != EAT_OPENGL || (params.presentMode == ISurface::EPM_MAILBOX) || (params.presentMode == ISurface::EPM_UNKNOWN))
        return nullptr;

    auto device = core::smart_refctd_ptr_static_cast<COpenGL_LogicalDevice>(logicalDevice);
    IGPUImage::SCreationParams imgci;
    imgci.arrayLayers = params.arrayLayers;
    imgci.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0);
    imgci.format = params.surfaceFormat.format;
    imgci.mipLevels = 1u;
    imgci.queueFamilyIndexCount = params.queueFamilyIndexCount;
    imgci.queueFamilyIndices = params.queueFamilyIndices;
    imgci.samples = asset::IImage::ESCF_1_BIT;
    imgci.type = asset::IImage::ET_2D;
    imgci.extent = asset::VkExtent3D{ params.width, params.height, 1u };
    imgci.usage = params.imageUsage;

    auto images = core::make_refctd_dynamic_array<ImagesArrayType>(params.minImageCount);
    for (auto& img_dst : (*images))
    {
        img_dst = device->createImage(IGPUImage::SCreationParams(imgci));
        auto mreq = img_dst->getMemoryReqs();
        mreq.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto imgMem = IDeviceMemoryAllocator::allocate(mreq, img_dst.get());
        if (!img_dst || !imgMem.isValid())
            return nullptr;
    }

    EGLConfig fbconfig = device->getEglConfig();
    auto glver = device->getGlVersion();

    // master context must not be current while creating a context with whom it will be sharing
    device->unbindMasterContext();
    EGLContext ctx = createGLContext(FunctionTableType::EGL_API_TYPE, device->getEgl(), glver.first, glver.second, fbconfig, m_threadHandler.glctx.ctx);
    auto sc = create(std::move(params), core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), device->getEgl(), std::move(images), device->getGlFeatures(), ctx, fbconfig, static_cast<COpenGLDebugCallback*>(device->getPhysicalDevice()->getDebugCallback()));
    if (!sc)
        return nullptr;
    // wait until swapchain's internal thread finish context creation
    sc->waitForInitComplete();
    // make master context (in logical device internal thread) again
    device->bindMasterContext();

    return sc;
}

template <typename FunctionTableType_, typename ThreadHandler>
nbl::video::ISwapchain::E_ACQUIRE_IMAGE_RESULT COpenGL_Swapchain<FunctionTableType_, ThreadHandler>::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override
{
    COpenGLSemaphore* glSem = IBackendObject::compatibility_cast<COpenGLSemaphore*>(semaphore, this);
    COpenGLFence* glFen = IBackendObject::compatibility_cast<COpenGLFence*>(fence, this);
    if (semaphore && !glSem)
        return EAIR_ERROR;
    if (fence && !glFen)
        return EAIR_ERROR;

    // TODO currently completely ignoring `timeout`

    ++m_imgIx;
    m_imgIx %= static_cast<uint32_t>(m_images->size());

    if (semaphore || fence)
    {
        core::smart_refctd_ptr<COpenGLSync> sync = m_threadHandler.getSyncForImgIx(m_imgIx);
        if (glSem)
            glSem->associateGLSync(core::smart_refctd_ptr(sync));
        if (glFen)
            glFen->associateGLSync(core::smart_refctd_ptr(sync));
    }

    assert(out_imgIx);
    out_imgIx[0] = m_imgIx;

    return EAIR_SUCCESS;
}

template <typename FunctionTableType_, typename ThreadHandler>
nbl::video::ISwapchain::E_PRESENT_RESULT COpenGL_Swapchain<FunctionTableType_, ThreadHandler>::present(IGPUQueue* queue, const SPresentInfo& info) override
{
    for (uint32_t i = 0u; i < info.waitSemaphoreCount; ++i)
    {
        if (getOriginDevice() != info.waitSemaphores[i]->getOriginDevice())
            return ISwapchain::EPR_ERROR;
    }

    bool retval = present(info.imgIndex, info.waitSemaphoreCount, info.waitSemaphores);
    return retval ? ISwapchain::EPR_SUCCESS : ISwapchain::EPR_ERROR;
}

template <typename FunctionTableType_, typename ThreadHandler>
void COpenGL_Swapchain<FunctionTableType_, ThreadHandler>::waitForInitComplete()
{
    m_threadHandler.waitForInitComplete();
}

}