#ifndef __NBL_C_OPENGL__SWAPCHAIN_H_INCLUDED__
#define __NBL_C_OPENGL__SWAPCHAIN_H_INCLUDED__

#include "nbl/video/ISwapchain.h"
#include "nbl/video/COpenGLFunctionTable.h"
#include "nbl/system/IThreadHandler.h"
#include "nbl/video/COpenGLImage.h"
#include "nbl/video/surface/ISurfaceGL.h"

namespace nbl {
namespace video
{

template <typename FunctionTableType_>
class COpenGL_Swapchain final : public ISwapchain
{
    static inline constexpr uint32_t MaxImages = 4u;

public:
    using ImagesArrayType = ISwapchain::images_array_t;
    using FunctionTableType = FunctionTableType_;

    // should be called by GL/GLES backend's impl of vkQueuePresentKHR
    void present(uint32_t _imgIx)
    {
        m_threadHandler.requestBlit(_imgIx);
    }

    static core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>> create(SCreationParams&& params, const egl::CEGL* _egl, ImagesArrayType&& images, COpenGLFeatureMap* _features, EGLContext _master, EGLConfig _config, EGLint _major, EGLint _minor)
    {
        if (!images || !images->size())
            return nullptr;
        if (images->size() < params.minImageCount)
            return nullptr;
        if (images->size() > MaxImages)
            return nullptr;

        auto extent = (*images)[0]->getCreationParameters().extent;
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

        return core::make_smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>>(std::move(params), _egl, std::move(images), _features, _master, _config, _major, _minor);
    }

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override
    {
        // TODO impl
        ++m_imgIx;
        m_imgIx %= static_cast<uint32_t>(m_images->size());
        out_imgIx[0] = m_imgIx;

        return EAIR_SUCCESS;
    }

protected:
    // images will be created in COpenGLLogicalDevice::createSwapchain
    COpenGL_Swapchain(SCreationParams&& params, const egl::CEGL* _egl, ImagesArrayType&& images, COpenGLFeatureMap* _features, EGLContext _master, EGLConfig _config, EGLint _major, EGLint _minor) :
        ISwapchain(std::move(params)),
        m_threadHandler(_egl, static_cast<ISurfaceGL*>(m_params.surface.get())->getInternalObject(), { images->begin(), images->end() }, _features, _master, _config, _major, _minor),
        m_thread(&CThreadHandler::thread, &m_threadHandler)
    {
        m_images = std::move(images);
    }

    ~COpenGL_Swapchain()
    {
        m_threadHandler.terminate(m_thread);
    }

private:
    using SThreadHandlerInternalState = FunctionTableType;
    class CThreadHandler final : public system::IThreadHandler<SThreadHandlerInternalState>
    {
    public:
        CThreadHandler(const egl::CEGL* _egl, EGLNativeWindowType _window, core::SRange<core::smart_refctd_ptr<IGPUImage>> _images, COpenGLFeatureMap* _features, EGLContext _master, EGLConfig _config, EGLint _major, EGLint _minor) :
            egl(_egl),
            masterCtx(_master), config(_config),
            major(_major), minor(_minor),
            window(_window),
            thisCtx(EGL_NO_CONTEXT), surface(EGL_NO_SURFACE),
            features(_features),
            images(_images)
        {
            assert(images.size() <= MaxImages);
        }

        void requestBlit(uint32_t _imgIx)
        {
            auto raii_handler = createRAIIDispatchHandler();

            needToBlit = true;
            imgIx = _imgIx;
        }

    protected:
        using base_t = system::IThreadHandler<SThreadHandlerInternalState>;

        SThreadHandlerInternalState init() override
        {
            egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

            const EGLint ctx_attributes[] = {
                EGL_CONTEXT_MAJOR_VERSION, major,
                EGL_CONTEXT_MINOR_VERSION, minor,
                //EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, // core is default for GL api (and would be wrong param in case of es)

                EGL_NONE
            };

            thisCtx = egl->call.peglCreateContext(egl->display, config, masterCtx, ctx_attributes);

            const EGLint surface_attributes[] = {
                EGL_GL_COLORSPACE, EGL_GL_COLORSPACE_SRGB,
                EGL_RENDER_BUFFER, EGL_BACK_BUFFER,

                EGL_NONE
            };
            surface = egl->call.peglCreateWindowSurface(egl->display, config, window, surface_attributes);


            egl->call.peglMakeCurrent(egl->display, surface, surface, thisCtx);

            const uint32_t fboCount = images.size();
            auto gl = SThreadHandlerInternalState(egl, features);

            gl.extGlCreateFramebuffers(fboCount, fbos);
            for (uint32_t i = 0u; i < fboCount; ++i)
            {
                GLuint fbo = fbos[i];
                auto& img = images.begin()[i];

                GLuint glimg = static_cast<COpenGLImage*>(img.get())->getOpenGLName();
                gl.extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, glimg, 0, GL_TEXTURE_2D);
                GLenum drawbuffer0 = GL_COLOR_ATTACHMENT0;
                gl.extGlNamedFramebufferDrawBuffers(fbo, 1, &drawbuffer0);
            }

            return gl;
        }

        void work(lock_t& lock, internal_state_t& gl) override
        {
            needToBlit = false;

            lock.unlock();

            const GLint w = images.begin()[imgIx]->getCreationParameters().extent.width;
            const GLint h = images.begin()[imgIx]->getCreationParameters().extent.height;
            // TODO
            // wait semaphores
            gl.extGlBlitNamedFramebuffer(fbos[imgIx], 0, 0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            egl->call.peglSwapBuffers(egl->display, surface);

            lock.lock();
        }

        void exit(internal_state_t& gl) override
        {
            gl.glFramebuffer.pglDeleteFramebuffers(images.size(), fbos);
            egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }

        bool wakeupPredicate() const override { return needToBlit || base_t::wakeupPredicate(); }
        bool continuePredicate() const override { return needToBlit && base_t::continuePredicate(); }

    private:
		const egl::CEGL* egl;
        EGLContext masterCtx;
        EGLConfig config;
        EGLint major, minor;
        EGLNativeWindowType window;
		EGLContext thisCtx;
		EGLSurface surface;
		COpenGLFeatureMap* features;
        core::SRange<core::smart_refctd_ptr<IGPUImage>> images;
        GLuint fbos[MaxImages]{};
        uint32_t imgIx = 0u;

        bool needToBlit = false;
    };

    CThreadHandler m_threadHandler;
    std::thread m_thread;
    uint32_t m_imgIx = 0u;
};

}
}

#endif
