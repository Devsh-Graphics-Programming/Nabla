#ifndef __NBL_C_OPENGL_QUEUE_H_INCLUDED__
#define __NBL_C_OPENGL_QUEUE_H_INCLUDED__

#include "nbl/video/IGPUQueue.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/video/COpenGLFunctionTable.h"
#include "nbl/system/IThreadHandler.h"

namespace nbl {
namespace video
{

class COpenGLQueue final : public IGPUQueue
{
        using SThreadHandlerInternalState = COpenGLFunctionTable;

        struct CThreadHandler final : public system::IThreadHandler<SThreadHandlerInternalState>
        {
        public:
            struct queue_element_t
            {
                SSubmitInfo submit;
            };

            void enqueue(queue_element_t&& e)
            {
                auto raii_handler = createRAIIDisptachHandler();

                q.push(std::move(e));
            }

            CThreadHandler(const egl::CEGL* _egl, COpenGLFeatureMap* _features, EGLContext _master, EGLConfig _config) :
                egl(_egl),
                thisCtx(EGL_NO_CONTEXT), pbuffer(EGL_NO_SURFACE),
                features(_features)
            {
                EGLint ctx_attributes[] = {
                    EGL_CONTEXT_MAJOR_VERSION, 4,
                    EGL_CONTEXT_MINOR_VERSION, 6,
                    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

                    EGL_NONE
                };

                do
                {
                    thisCtx = eglCreateContext(_egl->display, _config, _master, ctx_attributes);
                    --ctx_attributes[3];
                } while (thisCtx == EGL_NO_CONTEXT && ctx_attributes[3] >= 3); // fail if cant create >=4.3 context
                ++ctx_attributes[3];

                // why not 1x1?
                const EGLint pbuffer_attributes[] = {
                    EGL_WIDTH, 128,
                    EGL_HEIGHT, 128,

                    EGL_NONE
                };
                pbuffer = _egl->call.peglCreatePbufferSurface(_egl->display, _config, pbuffer_attributes);
            }

        protected:
            using base_t = system::IThreadHandler<SThreadHandlerInternalState>;

            SThreadHandlerInternalState init() override
            {
                return SThreadHandlerInternalState(&egl->call, features);
            }

            void work(lock_t& lock, internal_state_t& state) override
            {
                // pop item from queue

                lock.unlock();

                // wait semaphores
                // submit commands to GPU
                // glFlush?
                // signal semaphores
                // glFinish

                lock.lock();
            }

            bool wakeupPredicate() const override { return q.size() || base_t::wakeupPredicate(); }
            bool continuePredicate() const override { return q.size() && base_t::continuePredicate(); }

        private:
            const egl::CEGL* egl;
            EGLContext thisCtx;
            EGLSurface pbuffer;
            COpenGLFeatureMap* features;

            core::queue<queue_element_t> q;
        };

    public:
        COpenGLQueue(const egl::CEGL* _egl, COpenGLFeatureMap* _features, EGLContext _masterCtx, EGLConfig _config, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority) :
            IGPUQueue(_famIx, _flags, _priority),
            threadHandler(_egl, _features, _masterCtx, _config),
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
        ~COpenGLQueue()
        {
            threadHandler.terminate(thread);
        }

    private:
        CThreadHandler threadHandler;
        std::thread thread;
};

}}

#endif