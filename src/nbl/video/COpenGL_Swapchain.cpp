
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

static inline constexpr uint32_t OpenGLFunctionTableSize = std::max(sizeof(COpenGLFunctionTable), sizeof(COpenGLESFunctionTable));
using SThreadHandlerInternalState = std::array<uint8_t, OpenGLFunctionTableSize>;

IOpenGL_FunctionTable* getFunctionTable(video::E_API_TYPE apiType, SThreadHandlerInternalState* internalState)
{
    if (apiType == video::EAT_OPENGL)
        return reinterpret_cast<COpenGLFunctionTable*>(internalState);
    else if (apiType == video::EAT_OPENGL_ES)
        return reinterpret_cast<COpenGLESFunctionTable*>(internalState);
    else assert(false);
}

class COpenGL_SwapchainThreadHandler final : public system::IThreadHandler<COpenGL_SwapchainThreadHandler, SThreadHandlerInternalState>
{
public:
    using base_t = system::IThreadHandler<COpenGL_SwapchainThreadHandler, SThreadHandlerInternalState>;

    IOpenGL_LogicalDevice* m_device;
    uint64_t m_masterContextCallsWaited;

    const egl::CEGL* egl;
    ISurface::E_PRESENT_MODE m_presentMode;
    const COpenGLFeatureMap* features;
    uint32_t imageCount;
    uint32_t imgWidth, imgHeight;
    core::smart_refctd_ptr<COpenGLSync> syncs[ISwapchain::MaxImages];
    COpenGLDebugCallback* m_dbgCb;
    std::array<uint32_t, ISwapchain::MaxImages> m_fbos;
    std::array<uint32_t, ISwapchain::MaxImages> m_textures;
    asset::E_FORMAT m_ourFormat;
    GLenum m_glFormat;

    bool needToBlit = false;

    EGLBoolean m_makeCurrentRes = EGL_FALSE;
    //std::condition_variable m_ctxCreatedCvar;

    COpenGL_SwapchainThreadHandler(const egl::CEGL* _egl,
        IOpenGL_LogicalDevice* dev,
        ISurface::E_PRESENT_MODE presentMode,
        uint32_t _imgCount, uint32_t _imgWidth, uint32_t _imgHeight,
        const COpenGLFeatureMap* _features,
        egl::CEGL::Context _glctx,
        COpenGLDebugCallback* _dbgCb,
        asset::E_FORMAT format
    ) : m_device(dev), m_masterContextCallsWaited(0),
        egl(_egl),
        m_presentMode(presentMode),
        glctx(_glctx),
        features(_features),
        imageCount(_imgCount),
        imgWidth(_imgWidth),
        imgHeight(_imgHeight),
        m_dbgCb(_dbgCb),
        m_ourFormat(format)
    {}

    void requestBlit(uint32_t _imgIx, uint32_t semCount, IGPUSemaphore* const* const sems)
    {
        auto raii_handler = base_t::createRAIIDispatchHandler();

        needToBlit = true;
        request.imgIx = _imgIx;
        request.semCount = semCount;
        request.sems = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<COpenGLSemaphore>>>(semCount);
        for (uint32_t i = 0u; i < semCount; ++i)
        {
            COpenGLSemaphore* sem = IBackendObject::device_compatibility_cast<COpenGLSemaphore*>(sems[i], m_device);
            request.sems->begin()[i] = core::smart_refctd_ptr<COpenGLSemaphore>(sem);
        }
    }

    core::smart_refctd_ptr<COpenGLSync> getSyncForImgIx(uint32_t imgix)
    {
        auto lk = base_t::createLock();

        return syncs[imgix];
    }

    core::smart_refctd_ptr<IGPUImage> createImageAsImported(uint32_t imgix, IGPUImage::SCreationParams&& params)
    {
        GLuint texture = m_textures[imgix];

        auto image = core::make_smart_refctd_ptr<COpenGLImage>(
            core::smart_refctd_ptr<const ILogicalDevice>(m_device),
            0, std::move(params), 
            m_glFormat,
            GL_TEXTURE_2D,
            m_textures[imgix]
        );

        return image;
    }

    egl::CEGL::Context glctx;

    struct SRequest
    {
        uint32_t imgIx = 0u;
        core::smart_refctd_dynamic_array<core::smart_refctd_ptr<COpenGLSemaphore>> sems;
        uint32_t semCount = 0;
    } request;

    void init(SThreadHandlerInternalState* state_ptr)
    {
        egl->call.peglBindAPI(m_device->getEGLAPI());

        EGLBoolean mcres = egl->call.peglMakeCurrent(egl->display, glctx.surface, glctx.surface, glctx.ctx);
        m_makeCurrentRes = mcres;
        //m_ctxCreatedCvar.notify_one();
        if (mcres != EGL_TRUE)
            return;

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

        if (m_device->getAPIType() == video::EAT_OPENGL)
            new (state_ptr) COpenGLFunctionTable(egl, features, core::smart_refctd_ptr<system::ILogger>(m_dbgCb->getLogger()));
        else if (m_device->getAPIType() == video::EAT_OPENGL_ES)
            new (state_ptr) COpenGLESFunctionTable(egl, features, core::smart_refctd_ptr<system::ILogger>(m_dbgCb->getLogger()));
        else assert(false);
        auto gl = getFunctionTable(m_device->getAPIType(), state_ptr);

#ifdef _NBL_DEBUG
        gl->glGeneral.pglEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        // TODO: debug message control (to exclude callback spam)
#endif
        if (m_dbgCb)
            gl->extGlDebugMessageCallback(m_dbgCb->m_callback, m_dbgCb);

        m_glFormat = getSizedOpenGLFormatFromOurFormat(gl, m_ourFormat);
        
        gl->glGeneral.pglEnable(IOpenGL_FunctionTable::FRAMEBUFFER_SRGB);
        gl->extGlCreateFramebuffers(imageCount, m_fbos.data());

        for (uint32_t i = 0; i < imageCount; i++)
        {
            GLuint fbo = m_fbos.data()[i];

            // Create texture
            gl->extGlCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &m_textures.data()[i]);

            // Init memory
            GLuint textureName = m_textures.data()[i];
            gl->extGlTextureStorage3D(
                textureName, GL_TEXTURE_2D_ARRAY, 1, m_glFormat,
                imgWidth, imgHeight, 1
            );

            // Create FBO
            gl->extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, textureName, 0, GL_TEXTURE_2D);
            GLenum drawbuffer0 = GL_COLOR_ATTACHMENT0;
            gl->extGlNamedFramebufferDrawBuffers(fbo, 1, &drawbuffer0);

            GLenum status = gl->extGlCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
            assert(status == GL_FRAMEBUFFER_COMPLETE);
        }


        for (uint32_t i = 0u; i < imageCount; ++i)
        {
            syncs[i] = core::make_smart_refctd_ptr<COpenGLSync>();
            syncs[i]->init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(m_device), gl, false);
        }

        gl->glGeneral.pglFinish();
    }

    void work(typename base_t::lock_t& lock, typename base_t::internal_state_t& state)
    {
        auto gl = getFunctionTable(m_device->getAPIType(), &state);
        needToBlit = false;

        const uint32_t imgix = request.imgIx;
        const uint32_t w = imgWidth;
        const uint32_t h = imgHeight;

        for (uint32_t i = 0u; i < request.semCount; ++i)
        {
            core::smart_refctd_ptr<COpenGLSemaphore>& sem = request.sems->begin()[i];
            sem->wait(gl);
        }

        // need to possibly wait for master context (image & view creation, etc.)
        // likely not needed anymore, leaving just to be sure
        m_masterContextCallsWaited = m_device->waitOnMasterContext(gl, m_masterContextCallsWaited);

        gl->extGlBlitNamedFramebuffer(m_fbos.begin()[imgix], 0, 0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        syncs[imgix] = core::make_smart_refctd_ptr<COpenGLSync>();
        syncs[imgix]->init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(m_device), gl, false);
        // swap buffers performs an implicit flush before swapping 
        // https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglSwapBuffers.xhtml
        egl->call.peglSwapBuffers(egl->display, glctx.surface);
    }

    void exit(SThreadHandlerInternalState* state)
    {
        egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        egl->call.peglDestroyContext(egl->display, glctx.ctx);
        egl->call.peglDestroySurface(egl->display, glctx.surface);
    }

    bool wakeupPredicate() const { return needToBlit; }
    bool continuePredicate() const { return needToBlit; }
};

template <typename FunctionTableType>
const void* COpenGL_Swapchain<FunctionTableType>::getNativeHandle() const { return &m_threadHandler->glctx; }

template <typename FunctionTableType>
core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>> COpenGL_Swapchain<FunctionTableType>::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
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

    EGLConfig fbconfig = device->getEglConfig();
    auto glver = device->getGlVersion();

    // master context must not be current while creating a context with whom it will be sharing
    device->unbindMasterContext();
    EGLContext ctx = device->createGLContext(FunctionTableType::EGL_API_TYPE, device->getEgl(), glver.first, glver.second, fbconfig, device->getEglContext());

    if (params.minImageCount > ISwapchain::MaxImages)
        return nullptr;

    auto egl = device->getEgl();
    egl->call.peglBindAPI(device->getEGLAPI());

    const EGLint surface_attributes[] = {
        EGL_RENDER_BUFFER, EGL_BACK_BUFFER,
        // EGL_GL_COLORSPACE is supported only for EGL 1.5 and later
        egl->version.minor >= 5 ? EGL_GL_COLORSPACE : EGL_NONE, EGL_GL_COLORSPACE_SRGB,

        EGL_NONE
    };

    auto surface = egl->call.peglCreateWindowSurface(egl->display, fbconfig, (EGLNativeWindowType)params.surface->getNativeWindowHandle(), surface_attributes);
    assert(surface != EGL_NO_SURFACE);

    std::unique_ptr<COpenGL_SwapchainThreadHandler> threadHandler(new COpenGL_SwapchainThreadHandler(
        egl, 
        device.get(), 
        params.presentMode, 
        params.minImageCount, params.width, params.height,
        device->getGlFeatures(), { ctx, surface }, 
        static_cast<COpenGLDebugCallback*>(device->getPhysicalDevice()->getDebugCallback()), 
        params.surfaceFormat.format
    ));

    auto* sc = new COpenGL_Swapchain<FunctionTableType>(
        std::move(params), std::move(device), params.minImageCount, std::move(imgci), std::move(threadHandler)
     );
     sc->m_threadHandler->start();
     if (!sc)
        return nullptr;
    // wait until swapchain's internal thread finish context creation
     sc->m_threadHandler->waitForInitComplete();
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

template <typename FunctionTableType>
nbl::video::ISwapchain::E_ACQUIRE_IMAGE_RESULT COpenGL_Swapchain<FunctionTableType>::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx)
{
    COpenGLSemaphore* glSem = IBackendObject::compatibility_cast<COpenGLSemaphore*>(semaphore, this);
    COpenGLFence* glFen = IBackendObject::compatibility_cast<COpenGLFence*>(fence, this);
    if (semaphore && !glSem)
        return EAIR_ERROR;
    if (fence && !glFen)
        return EAIR_ERROR;

    // TODO currently completely ignoring `timeout`

    ++m_imgIx;
    m_imgIx %= static_cast<uint32_t>(m_imageCount);

    if (semaphore || fence)
    {
        core::smart_refctd_ptr<COpenGLSync> sync = m_threadHandler->getSyncForImgIx(m_imgIx);
        if (glSem)
            glSem->associateGLSync(core::smart_refctd_ptr(sync));
        if (glFen)
            glFen->associateGLSync(core::smart_refctd_ptr(sync));
    }

    assert(out_imgIx);
    out_imgIx[0] = m_imgIx;

    return EAIR_SUCCESS;
}

template <typename FunctionTableType>
nbl::video::ISwapchain::E_PRESENT_RESULT COpenGL_Swapchain<FunctionTableType>::present(IGPUQueue* queue, const SPresentInfo& info)
{
    for (uint32_t i = 0u; i < info.waitSemaphoreCount; ++i)
    {
        if (getOriginDevice() != info.waitSemaphores[i]->getOriginDevice())
            return ISwapchain::EPR_ERROR;
    }

    uint32_t _imgIx = info.imgIndex;
    uint32_t semCount = info.waitSemaphoreCount;
    IGPUSemaphore* const* const sems = info.waitSemaphores;

    if (_imgIx >= m_params.minImageCount)
        return ISwapchain::EPR_ERROR;
    for (uint32_t i = 0u; i < semCount; ++i)
    {
        if (!this->isCompatibleDevicewise(sems[i]))
            return ISwapchain::EPR_ERROR;
    }
    m_threadHandler->requestBlit(_imgIx, semCount, sems);

    return ISwapchain::EPR_SUCCESS;
}

template <typename FunctionTableType>
core::smart_refctd_ptr<IGPUImage> COpenGL_Swapchain<FunctionTableType>::createImage(const uint32_t imageIndex)
{
    if (!setImageExists(imageIndex))
        return nullptr;

    auto& device = m_threadHandler->m_device;
    auto imgCreationParams = std::move(m_imgCreationParams);
    std::unique_ptr<CCleanupSwapchainReference> swapchainRef(new CCleanupSwapchainReference{});
    swapchainRef->m_swapchain = core::smart_refctd_ptr<ISwapchain>(this);
    swapchainRef->m_imageIndex = imageIndex;

    imgCreationParams.preDestroyCleanup = std::unique_ptr<ICleanup>(swapchainRef.release());
    imgCreationParams.importedHandle = true;

    return m_threadHandler->createImageAsImported(imageIndex, std::move(imgCreationParams));
}

template core::smart_refctd_ptr<COpenGLSwapchain> COpenGLSwapchain::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);
template core::smart_refctd_ptr<COpenGLESSwapchain> COpenGLESSwapchain::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

}