
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
static_assert(OpenGLFunctionTableSize >= sizeof(COpenGLFunctionTable));
static_assert(OpenGLFunctionTableSize >= sizeof(COpenGLESFunctionTable));

IOpenGL_FunctionTable* getFunctionPointer(video::E_API_TYPE apiType, SThreadHandlerInternalState* internalState)
{  
    if (apiType == video::EAT_OPENGL)
        return reinterpret_cast<COpenGLFunctionTable*>(internalState);
    else if (apiType == video::EAT_OPENGL_ES)
        return reinterpret_cast<COpenGLESFunctionTable*>(internalState);
    else assert(false);
}

COpenGL_SwapchainThreadHandler::COpenGL_SwapchainThreadHandler(const egl::CEGL* _egl,
    core::smart_refctd_ptr<IOpenGL_LogicalDevice> dev,
    const void* _window,
    ISurface::E_PRESENT_MODE presentMode,
    core::SRange<core::smart_refctd_ptr<IGPUImage>> _images,
    const COpenGLFeatureMap* _features,
    EGLContext _ctx,
    EGLConfig _config,
    COpenGLDebugCallback* _dbgCb
) : m_device(dev), m_masterContextCallsWaited(0),
    egl(_egl),
    m_presentMode(presentMode),
    glctx{ _ctx,EGL_NO_SURFACE },
    features(_features),
    images(_images),
    m_dbgCb(_dbgCb)
{
    assert(images.size() <= MaxImages);
    _egl->call.peglBindAPI(m_device->getEGLAPI());

    const EGLint surface_attributes[] = {
        EGL_RENDER_BUFFER, EGL_BACK_BUFFER,
        // EGL_GL_COLORSPACE is supported only for EGL 1.5 and later
        _egl->version.minor >= 5 ? EGL_GL_COLORSPACE : EGL_NONE, EGL_GL_COLORSPACE_SRGB,

        EGL_NONE
    };

    glctx.surface = _egl->call.peglCreateWindowSurface(_egl->display, _config, (EGLNativeWindowType)_window, surface_attributes);
    assert(glctx.surface != EGL_NO_SURFACE);

    base_t::start();
}

void COpenGL_SwapchainThreadHandler::requestBlit(uint32_t _imgIx, uint32_t semCount, IGPUSemaphore* const* const sems)
{
    auto raii_handler = base_t::createRAIIDispatchHandler();

    needToBlit = true;
    request.imgIx = _imgIx;
    request.semCount = semCount;
    request.sems.clear();
    if (request.sems.capacity() < semCount)
        request.sems.reserve(semCount);
    for (uint32_t i = 0u; i < semCount; ++i)
    {
        COpenGLSemaphore* sem = IBackendObject::device_compatibility_cast<COpenGLSemaphore*>(sems[i], m_device.get());
        request.sems.push_back(core::smart_refctd_ptr<COpenGLSemaphore>(sem));
    }
}

core::smart_refctd_ptr<COpenGLSync> COpenGL_SwapchainThreadHandler::getSyncForImgIx(uint32_t imgix)
{
    auto lk = base_t::createLock();

    return syncs[imgix];
}

void COpenGL_SwapchainThreadHandler::init(SThreadHandlerInternalState* state_ptr)
{

    egl->call.peglBindAPI(m_device->getEGLAPI());

    EGLBoolean mcres = egl->call.peglMakeCurrent(egl->display, glctx.surface, glctx.surface, glctx.ctx);
    m_makeCurrentRes = mcres;
    if (mcres != EGL_TRUE)
        return;

    m_ctxCreatedCvar.notify_one();

    switch (m_presentMode)
    {
    case ISurface::EPM_IMMEDIATE:
        egl->call.peglSwapInterval(egl->display, 0);
        break;
    case ISurface::EPM_FIFO:
        egl->call.peglSwapInterval(egl->display, 1);
        break;
    case ISurface::EPM_FIFO_RELAXED:
        egl->call.peglSwapInterval(egl->display, -1);
        break;
    }

    const uint32_t fboCount = images.size();
    if (m_device->getAPIType() == video::EAT_OPENGL)
        new (state_ptr) COpenGLFunctionTable(egl, features, core::smart_refctd_ptr<system::ILogger>(m_dbgCb->getLogger()));
    else if (m_device->getAPIType() == video::EAT_OPENGL_ES)
        new (state_ptr) COpenGLESFunctionTable(egl, features, core::smart_refctd_ptr<system::ILogger>(m_dbgCb->getLogger()));
    else assert(false);
    auto gl = getFunctionPointer(m_device->getAPIType(), state_ptr);

    gl->glTexture.pglGenTextures(images.size(), m_texViews.data());
    for (int i = 0; i < images.size(); i++)
    {
        auto& img = images.begin()[i];
        GLuint texture = m_texViews[i];
        GLuint origtexture = IBackendObject::device_compatibility_cast<COpenGLImage*>(img.get(), m_device.get())->getOpenGLName();
        GLenum format = IBackendObject::device_compatibility_cast<COpenGLImage*>(img.get(), m_device.get())->getOpenGLSizedFormat();
        gl->extGlTextureView(texture, GL_TEXTURE_2D, origtexture, format, 0, 1, 0, 1);
    }

#ifdef _NBL_DEBUG
    gl->glGeneral.pglEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    // TODO: debug message control (to exclude callback spam)
#endif
    if (m_dbgCb)
        gl->extGlDebugMessageCallback(m_dbgCb->m_callback, m_dbgCb);

    gl->glGeneral.pglEnable(IOpenGL_FunctionTable::FRAMEBUFFER_SRGB);

    gl->extGlCreateFramebuffers(fboCount, fbos);
    for (uint32_t i = 0u; i < fboCount; ++i)
    {
        GLuint fbo = fbos[i];
        auto& img = images.begin()[i];
        gl->extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, m_texViews[i], 0, GL_TEXTURE_2D);
        GLenum drawbuffer0 = GL_COLOR_ATTACHMENT0;
        gl->extGlNamedFramebufferDrawBuffers(fbo, 1, &drawbuffer0);

        GLenum status = gl->extGlCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
        assert(status == GL_FRAMEBUFFER_COMPLETE);
    }
    for (uint32_t i = 0u; i < fboCount; ++i)
    {
        syncs[i] = core::make_smart_refctd_ptr<COpenGLSync>();
        syncs[i]->init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(m_device), gl, false);
    }

    gl->glGeneral.pglFinish();
}

void COpenGL_SwapchainThreadHandler::work(typename base_t::lock_t& lock, typename base_t::internal_state_t& state)
{
    needToBlit = false;

    const uint32_t imgix = request.imgIx;
    const uint32_t w = images.begin()[imgix]->getCreationParameters().extent.width;
    const uint32_t h = images.begin()[imgix]->getCreationParameters().extent.height;

    auto gl = getFunctionPointer(m_device->getAPIType(), &state);
    for (uint32_t i = 0u; i < request.semCount; ++i)
    {
        core::smart_refctd_ptr<COpenGLSemaphore>& sem = request.sems[i];
        sem->wait(gl);
    }

    // need to possibly wait for master context (image & view creation, etc.)
    m_masterContextCallsWaited = m_device->waitOnMasterContext(gl, m_masterContextCallsWaited);

    gl->extGlBlitNamedFramebuffer(fbos[imgix], 0, 0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    syncs[imgix] = core::make_smart_refctd_ptr<COpenGLSync>();
    syncs[imgix]->init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(m_device), gl, false);
    // swap buffers performs an implicit flush before swapping 
    // https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglSwapBuffers.xhtml
    egl->call.peglSwapBuffers(egl->display, glctx.surface);
}

void COpenGL_SwapchainThreadHandler::exit(SThreadHandlerInternalState* state)
{
    auto gl = getFunctionPointer(m_device->getAPIType(), state);
    gl->glFramebuffer.pglDeleteFramebuffers(images.size(), fbos);
    gl->glTexture.pglDeleteTextures(images.size(), m_texViews.data());
    gl->glGeneral.pglFinish();

    gl->~IOpenGL_FunctionTable();

    egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    egl->call.peglDestroyContext(egl->display, glctx.ctx);
    egl->call.peglDestroySurface(egl->display, glctx.surface);
}

template <typename FunctionTableType_>
COpenGL_Swapchain<FunctionTableType_>::COpenGL_Swapchain<FunctionTableType_>(
    SCreationParams&& params,
    core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev,
    const egl::CEGL* _egl,
    ImagesArrayType&& images,
    const COpenGLFeatureMap* _features,
    EGLContext _ctx,
    EGLConfig _config,
    COpenGLDebugCallback* _dbgCb
) : ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>(dev), std::move(params), std::move(images)),
    m_threadHandler(
        _egl, dev, m_params.surface->getNativeWindowHandle(), m_params.presentMode, { m_images->begin(),m_images->end() }, _features, _ctx, _config, _dbgCb
    )
{}

template <typename FunctionTableType_>
core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType_>> COpenGL_Swapchain<FunctionTableType_>::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{
    if (params.surface->getAPIType() != EAT_OPENGL || (params.presentMode == ISurface::EPM_MAILBOX) || (params.presentMode == ISurface::EPM_UNKNOWN))
        return nullptr;

    auto device = core::smart_refctd_ptr_static_cast<IOpenGL_LogicalDevice>(logicalDevice);
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
        auto imgMem = device->allocate(mreq, img_dst.get());
        if (!img_dst || !imgMem.isValid())
            return nullptr;
    }

    EGLConfig fbconfig = device->getEglConfig();
    auto glver = device->getGlVersion();

    // master context must not be current while creating a context with whom it will be sharing
    device->unbindMasterContext();
    EGLContext ctx = device->createGLContext(FunctionTableType::EGL_API_TYPE, device->getEgl(), glver.first, glver.second, fbconfig, device->getEglContext());

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

    auto* sc = new COpenGL_Swapchain<FunctionTableType>(
        std::move(params), std::move(device), device->getEgl(), std::move(images), device->getGlFeatures(), ctx, 
        fbconfig, static_cast<COpenGLDebugCallback*>(device->getPhysicalDevice()->getDebugCallback()));
    sc->m_threadHandler.request.sems.reserve(50);
    
     if (!sc)
        return nullptr;
    // wait until swapchain's internal thread finish context creation
     sc->m_threadHandler.waitForInitComplete();
    // make master context (in logical device internal thread) again
    device->bindMasterContext();

    return core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>>(sc, core::dont_grab);
}

core::smart_refctd_ptr<COpenGLSwapchain> createGLSwapchain(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{
    return COpenGLSwapchain::create(std::move(logicalDevice), std::move(params));
}

core::smart_refctd_ptr<COpenGLESSwapchain> createGLESSwapchain(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{
    return COpenGLESSwapchain::create(std::move(logicalDevice), std::move(params));
}

template <typename FunctionTableType_>
nbl::video::ISwapchain::E_ACQUIRE_IMAGE_RESULT COpenGL_Swapchain<FunctionTableType_>::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx)
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

template <typename FunctionTableType_>
nbl::video::ISwapchain::E_PRESENT_RESULT COpenGL_Swapchain<FunctionTableType_>::present(IGPUQueue* queue, const SPresentInfo& info)
{
    for (uint32_t i = 0u; i < info.waitSemaphoreCount; ++i)
    {
        if (getOriginDevice() != info.waitSemaphores[i]->getOriginDevice())
            return ISwapchain::EPR_ERROR;
    }

    bool retval = present(info.imgIndex, info.waitSemaphoreCount, info.waitSemaphores);
    return retval ? ISwapchain::EPR_SUCCESS : ISwapchain::EPR_ERROR;
}

}