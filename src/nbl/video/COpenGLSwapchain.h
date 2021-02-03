#ifndef __NBL_C_OPENGL_SWAPCHAIN_H_INCLUDED__
#define __NBL_C_OPENGL_SWAPCHAIN_H_INCLUDED__

#include "nbl/video/ISwapchain.h"
#include "nbl/video/COpenGLFunctionTable.h"
#include "nbl/system/IThreadHandler.h"
#include "nbl/video/COpenGLImage.h"
#include "nbl/video/surfaces/ISurfaceGL.h"

namespace nbl {
namespace video
{

class COpenGLSwapchain final : public ISwapchain
{
public:
    // images will be created in COpenGLLogicalDevice::createSwapchain
    COpenGLSwapchain(SCreationParams&& params, const egl::CEGL* _egl, images_array_t&& images, COpenGLFeatureMap* _features, EGLContext _master, EGLConfig _config, EGLint _major, EGLint _minor) :
        ISwapchain(std::move(params)),
        m_threadHandler(_egl, static_cast<ISurfaceGL*>(m_params.surface.get())->getInternalObject(), {images->begin(), images->end()}, _features, _master, _config, _major, _minor),
        m_thread(&CThreadHandler::thread, &m_threadHandler)
    {
        m_images = std::move(images);
    }

    // should be called by GL/GLES backend's impl of vkQueuePresentKHR
    void present(uint32_t _imgIx)
    {
        m_threadHandler.requestBlit(_imgIx);
    }

protected:
    ~COpenGLSwapchain()
    {
        m_threadHandler.terminate(m_thread);
    }

private:
    using SThreadHandlerInternalState = COpenGLFunctionTable;
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

        }

        void requestBlit(uint32_t _imgIx)
        {
            auto raii_handler = createRAIIDisptachHandler();

            needToBlit = true;
            imgIx = _imgIx;
        }

    protected:
        using base_t = system::IThreadHandler<SThreadHandlerInternalState>;

        SThreadHandlerInternalState init() override
        {
            egl->call.peglBindAPI(EGL_OPENGL_API);

            const EGLint ctx_attributes[] = {
                EGL_CONTEXT_MAJOR_VERSION, major,
                EGL_CONTEXT_MINOR_VERSION, minor,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

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
            auto gl = SThreadHandlerInternalState(&egl->call, features);

            gl.extGlCreateFramebuffers(fboCount, fbos);
            for (uint32_t i = 0u; i < fboCount; ++i)
            {
                GLuint fbo = fbos[i];
                auto& img = images.begin()[i];
                GLuint glimg = static_cast<COpenGLImage*>(img.get())->getOpenGLName();
                gl.extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, glimg, 0);
            }

            return gl;
        }

        void work(lock_t& lock, internal_state_t& gl) override
        {
            needToBlit = false;

            lock.unlock();

            const GLint w = images.begin()[0]->getCreationParameters().extent.width;
            const GLint h = images.begin()[0]->getCreationParameters().extent.height;
            // TODO
            // wait semaphores
            gl.extGlBlitNamedFramebuffer(0, fbos[imgIx], 0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
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
        inline constexpr static uint32_t MaxImages = 4u;

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
};

}
}

#endif
