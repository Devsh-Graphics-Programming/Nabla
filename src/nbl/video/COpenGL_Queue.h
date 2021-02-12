#ifndef __NBL_C_OPENGL__QUEUE_H_INCLUDED__
#define __NBL_C_OPENGL__QUEUE_H_INCLUDED__

#include "nbl/video/IGPUQueue.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/video/CEGL.h"
#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl {
namespace video
{

template <typename FunctionTableType_>
class COpenGL_Queue final : public IGPUQueue
{
    public:
        using FunctionTableType = FunctionTableType_;
        using FeaturesType = typename FunctionTableType::features_t;

    private:
        using ThreadHandlerInternalStateType = FunctionTableType;
        using QueueElementType = SSubmitInfo; // TODO might want to change to array of SSubmitInfo + fence

        struct CThreadHandler final : public system::IAsyncQueueDispatcher<QueueElementType, ThreadHandlerInternalStateType>
        {
        public:
            CThreadHandler(const egl::CEGL* _egl, FeaturesType* _features, EGLContext _master, EGLConfig _config, EGLint _major, EGLint _minor) :
                egl(_egl),
                masterCtx(_master), config(_config),
                major(_major), minor(_minor),
                thisCtx(EGL_NO_CONTEXT), pbuffer(EGL_NO_SURFACE),
                features(_features)
            {

            }

        protected:
            using base_t = system::IAsyncQueueDispatcher<QueueElementType, ThreadHandlerInternalStateType>;

            ThreadHandlerInternalStateType init() override
            {
                egl->call.peglBindAPI(EGL_OPENGL_API);

                const EGLint ctx_attributes[] = {
                    EGL_CONTEXT_MAJOR_VERSION, major,
                    EGL_CONTEXT_MINOR_VERSION, minor,
                    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

                    EGL_NONE
                };

                thisCtx = egl->call.peglCreateContext(egl->display, config, masterCtx, ctx_attributes);

                // why not 1x1?
                const EGLint pbuffer_attributes[] = {
                    EGL_WIDTH, 128,
                    EGL_HEIGHT, 128,

                    EGL_NONE
                };
                pbuffer = egl->call.peglCreatePbufferSurface(egl->display, config, pbuffer_attributes);

                egl->call.peglMakeCurrent(egl->display, pbuffer, pbuffer, thisCtx);

                return ThreadHandlerInternalStateType(&egl->call, features);
            }

            void processElement(ThreadHandlerInternalStateType& _gl, queue_element_t&& e) const override
            {
                static_assert(std::is_same_v<ThreadHandlerInternalStateType, FunctionTableType>);
                // a cast to common base so that intellisense knows function set (can and should be removed after code gets written)
                IOpenGL_FunctionTable& gl = static_cast<IOpenGL_FunctionTable&>(_gl);
                // wait semaphores
                // TODO glMemoryBarrier() corresponding to _submit.pWaitDstStageMask[i]
                // submit commands to GPU
                // glFlush?
                // signal semaphores
                // glFinish
            }

            void exit(internal_state_t& gl) override
            {
                egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // detach ctx from thread
                egl->call.peglDestroyContext(egl->display, thisCtx);
            }

        private:
            const egl::CEGL* egl;
            EGLContext masterCtx;
            EGLConfig config;
            EGLint major, minor;
            EGLContext thisCtx;
            EGLSurface pbuffer;
            FeaturesType* features;
        };

    public:
        COpenGL_Queue(const egl::CEGL* _egl, FeaturesType* _features, EGLContext _masterCtx, EGLConfig _config, EGLint _gl_major, EGLint _gl_minor, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority) :
            IGPUQueue(_famIx, _flags, _priority),
            threadHandler(_egl, _features, _masterCtx, _config, _gl_major, _gl_minor),
            thread(&CThreadHandler::thread, &threadHandler)
        {

        }

        void submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override
        {
            core::smart_refctd_ptr<COpenGLSync> lastSync;
            for (uint32_t j = 0u; j < _count; ++j) // move to OpenGL backend
            {
                auto& submit = _submits[j];
                for (uint32_t i = 0u; i < submit.waitSemaphoreCount; ++i)
                {
                    auto* sem = static_cast<COpenGLSemaphore*>(submit.pWaitSemaphores[i]);
                    sem->wait();
                }
                
                for (uint32_t i = 0u; i < submit.commandBufferCount; ++i)
                {
                    auto* cmdbuf = submit.commandBuffers[i];

                    // TODO glMemoryBarrier() corresponding to _submit.pWaitDstStageMask[i]

                    //cmdbuf->setState(IGPUCommandBuffer::ES_PENDING); // TODO: setState() is inaccessible
                    // TODO actually submit the gl work in the command buffer

                    /*Once execution of all submissions of a command buffer complete, it moves from the pending state, back to the executable state.
                    If a command buffer was recorded with the VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT flag, it instead moves to the invalid state.
                    */
                }

                lastSync = core::make_smart_refctd_ptr<COpenGLSync>();
                //m_gl->glGeneral.pglFlush();
                for (uint32_t i = 0u; i < submit.signalSemaphoreCount; ++i)
                {
                    auto* sem = static_cast<COpenGLSemaphore*>(submit.pSignalSemaphores[i]);
                    sem->signal(core::smart_refctd_ptr(lastSync));
                }
            }

            if (_fence)
                static_cast<COpenGLFence*>(_fence)->associateGLSync(std::move(lastSync));
        }

    protected:
        ~COpenGL_Queue()
        {
            threadHandler.terminate(thread);
        }

    private:
        CThreadHandler threadHandler;
        std::thread thread;
};

}}

#endif