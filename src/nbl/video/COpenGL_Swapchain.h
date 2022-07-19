#ifndef __NBL_VIDEO_C_OPENGL__SWAPCHAIN_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL__SWAPCHAIN_H_INCLUDED__

#include "nbl/system/IThreadHandler.h"

#include "nbl/video/ISwapchain.h"
#include "nbl/video/surface/CSurfaceGL.h"

#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/COpenGLImage.h"

namespace nbl::video
{

class IOpenGL_LogicalDevice;

template <typename FunctionTableType_>
class COpenGL_Swapchain final : public ISwapchain
{
    static inline constexpr uint32_t MaxImages = 4u;
public:
    using ImagesArrayType = ISwapchain::images_array_t;
    using FunctionTableType = FunctionTableType_;

    // should be called by GL/GLES backend's impl of vkQueuePresentKHR
    inline bool present(uint32_t _imgIx, uint32_t semCount, IGPUSemaphore*const *const sems)
    {
        if (_imgIx >= m_params.minImageCount)
            return false;
        for (uint32_t i = 0u; i < semCount; ++i)
        {
            if (!this->isCompatibleDevicewise(sems[i]))
                return false;
        }
        m_threadHandler.requestBlit(_imgIx, semCount, sems);

        return true;
    }

    static core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>> create(SCreationParams&& params,
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

        auto* sc = new COpenGL_Swapchain<FunctionTableType>(std::move(params),std::move(dev),_egl,std::move(images),_features,_ctx,_config,_dbgCb);
        return core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>>(sc,core::dont_grab);
    }

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override
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

    void waitForInitComplete()
    {
        m_threadHandler.waitForInitComplete();
    }

    virtual const void* getNativeHandle() const override {return &m_threadHandler.glctx;}

protected:
    // images will be created in COpenGLLogicalDevice::createSwapchain
    COpenGL_Swapchain(
        SCreationParams&& params,
        core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev,
        const egl::CEGL* _egl,
        ImagesArrayType&& images,
        const COpenGLFeatureMap* _features,
        EGLContext _ctx,
        EGLConfig _config,
        COpenGLDebugCallback* _dbgCb
    ) : ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>(dev),std::move(params),std::move(images)),
        m_threadHandler(
            _egl,dev.get(),m_params.surface->getNativeWindowHandle(),m_params.presentMode,{m_images->begin(),m_images->end()},_features,_ctx,_config,_dbgCb
        )
    {}

private:
    using SThreadHandlerInternalState = FunctionTableType;
    class CThreadHandler final : public system::IThreadHandler<CThreadHandler, SThreadHandlerInternalState>
    {
        using base_t = system::IThreadHandler<CThreadHandler, SThreadHandlerInternalState>;
        friend base_t;

    public:
        CThreadHandler(const egl::CEGL* _egl,
            IOpenGL_LogicalDevice* dev,
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
            glctx{_ctx,EGL_NO_SURFACE},
            features(_features),
            images(_images),
            m_dbgCb(_dbgCb)
        {
            assert(images.size() <= MaxImages);
            _egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

            const EGLint surface_attributes[] = {
                EGL_RENDER_BUFFER, EGL_BACK_BUFFER,
                // EGL_GL_COLORSPACE is supported only for EGL 1.5 and later
                _egl->version.minor>=5 ? EGL_GL_COLORSPACE : EGL_NONE, EGL_GL_COLORSPACE_SRGB,

                EGL_NONE
            };

            glctx.surface = _egl->call.peglCreateWindowSurface(_egl->display, _config, (EGLNativeWindowType)_window, surface_attributes);
            assert(glctx.surface != EGL_NO_SURFACE);

            base_t::start();
        }

        void requestBlit(uint32_t _imgIx, uint32_t semCount, IGPUSemaphore*const *const sems)
        {
            auto raii_handler = base_t::createRAIIDispatchHandler();

            needToBlit = true;
            request.imgIx = _imgIx;
            request.semCount = semCount;
            request.sems.clear();
            if (request.sems.capacity()<semCount)
                request.sems.reserve(semCount);
            for (uint32_t i = 0u; i < semCount; ++i)
            {
                COpenGLSemaphore* sem = IBackendObject::device_compatibility_cast<COpenGLSemaphore*>(sems[i], m_device);
                request.sems.push_back(core::smart_refctd_ptr<COpenGLSemaphore>(sem));
            }
        }

        core::smart_refctd_ptr<COpenGLSync> getSyncForImgIx(uint32_t imgix)
        {
            auto lk = base_t::createLock();

            return syncs[imgix];
        }

        egl::CEGL::Context glctx;
    protected:

        void init(SThreadHandlerInternalState* state_ptr)
        {
            egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

            EGLBoolean mcres = egl->call.peglMakeCurrent(egl->display, glctx.surface, glctx.surface, glctx.ctx);
            assert(mcres == EGL_TRUE);

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
            new (state_ptr) SThreadHandlerInternalState(egl,features,core::smart_refctd_ptr<system::ILogger>(m_dbgCb->getLogger()));
            auto& gl = state_ptr[0];

            static_cast<IOpenGL_FunctionTable&>(gl).glTexture.pglGenTextures(images.size(), m_texViews.data());
            for (int i = 0; i < images.size(); i++)
            {
                auto& img = images.begin()[i];
                GLuint texture = m_texViews[i];
                GLuint origtexture = IBackendObject::device_compatibility_cast<COpenGLImage*>(img.get(), m_device)->getOpenGLName();
                GLenum format = IBackendObject::device_compatibility_cast<COpenGLImage*>(img.get(), m_device)->getOpenGLSizedFormat();
                static_cast<IOpenGL_FunctionTable&>(gl).extGlTextureView(texture, GL_TEXTURE_2D, origtexture, format, 0, 1, 0, 1);
            }
            
            #ifdef _NBL_DEBUG
            gl.glGeneral.pglEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            // TODO: debug message control (to exclude callback spam)
            #endif
            if (m_dbgCb)
                gl.extGlDebugMessageCallback(m_dbgCb->m_callback,m_dbgCb);

            gl.glGeneral.pglEnable(IOpenGL_FunctionTable::FRAMEBUFFER_SRGB);

            gl.extGlCreateFramebuffers(fboCount, fbos);
            for (uint32_t i = 0u; i < fboCount; ++i)
            {
                GLuint fbo = fbos[i];
                auto& img = images.begin()[i];
                gl.extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, m_texViews[i], 0, GL_TEXTURE_2D);
                GLenum drawbuffer0 = GL_COLOR_ATTACHMENT0;
                gl.extGlNamedFramebufferDrawBuffers(fbo, 1, &drawbuffer0);

                GLenum status = gl.extGlCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
                assert(status == GL_FRAMEBUFFER_COMPLETE);
            }
            for (uint32_t i = 0u; i < fboCount; ++i)
            {
                syncs[i] = core::make_smart_refctd_ptr<COpenGLSync>();
                syncs[i]->init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(m_device), &gl, false);
            }

            gl.glGeneral.pglFinish();
        }

        void work(typename base_t::lock_t& lock, typename base_t::internal_state_t& gl)
        {
            needToBlit = false;

            const uint32_t imgix = request.imgIx;
            const GLint w = images.begin()[imgix]->getCreationParameters().extent.width;
            const GLint h = images.begin()[imgix]->getCreationParameters().extent.height;

            for (uint32_t i = 0u; i < request.semCount; ++i)
            {
                core::smart_refctd_ptr<COpenGLSemaphore>& sem = request.sems[i];
                sem->wait(&gl);
            }

            // need to possibly wait for master context (image & view creation, etc.)
            m_masterContextCallsWaited = m_device->waitOnMasterContext(gl,m_masterContextCallsWaited);

            gl.extGlBlitNamedFramebuffer(fbos[imgix], 0, 0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            syncs[imgix] = core::make_smart_refctd_ptr<COpenGLSync>();
            syncs[imgix]->init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(m_device), &gl, false);
            // swap buffers performs an implicit flush before swapping 
            // https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglSwapBuffers.xhtml
            egl->call.peglSwapBuffers(egl->display, glctx.surface);
        }

        void exit(SThreadHandlerInternalState* gl)
        {
            gl->glFramebuffer.pglDeleteFramebuffers(images.size(), fbos);
            gl->glTexture.pglDeleteTextures(images.size(), m_texViews.data());
            gl->glGeneral.pglFinish();

            gl->~SThreadHandlerInternalState();

            egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            egl->call.peglDestroyContext(egl->display, glctx.ctx);
            egl->call.peglDestroySurface(egl->display, glctx.surface);
        }

        bool wakeupPredicate() const { return needToBlit; }
        bool continuePredicate() const { return needToBlit; }

    private:
        IOpenGL_LogicalDevice* m_device;
        uint64_t m_masterContextCallsWaited;

		const egl::CEGL* egl;
        ISurface::E_PRESENT_MODE m_presentMode;
		const COpenGLFeatureMap* features;
        core::SRange<core::smart_refctd_ptr<IGPUImage>> images;
        GLuint fbos[MaxImages]{};
        core::smart_refctd_ptr<COpenGLSync> syncs[MaxImages];
        COpenGLDebugCallback* m_dbgCb;
        std::array<GLuint,MaxImages> m_texViews;
        struct SRequest
        {
            SRequest() { sems.reserve(50); }

            uint32_t imgIx = 0u;
            core::vector<core::smart_refctd_ptr<COpenGLSemaphore>> sems;
            uint32_t semCount = 0;
        } request;

        bool needToBlit = false;

        EGLBoolean m_makeCurrentRes = EGL_FALSE;
        std::condition_variable m_ctxCreatedCvar;
    };

    CThreadHandler m_threadHandler;
    uint32_t m_imgIx = 0u;
};

}


#include "nbl/video/COpenGLFunctionTable.h"
#include "nbl/video/COpenGLESFunctionTable.h"

namespace nbl::video
{

using COpenGLSwapchain = COpenGL_Swapchain<COpenGLFunctionTable>;
using COpenGLESSwapchain = COpenGL_Swapchain<COpenGLESFunctionTable>;

}

#endif
