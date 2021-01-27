#ifndef __NBL_C_OPENGL_QUEUE_H_INCLUDED__
#define __NBL_C_OPENGL_QUEUE_H_INCLUDED__

#include "nbl/video/IGPUQueue.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/video/COpenGLFunctionTable.h"

namespace nbl {
namespace video
{

class COpenGLQueue final : public IGPUQueue
{
    public:
        COpenGLQueue(COpenGLFunctionTable* _gl, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority) :
            IGPUQueue(_famIx, _flags, _priority), m_gl(_gl)
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
                m_gl->glGeneral.pglFlush();
                for (uint32_t i = 0u; i < submit.signalSemaphoreCount; ++i)
                {
                    auto* sem = static_cast<COpenGLSemaphore*>(submit.pSignalSemaphores[i]);
                    sem->signal(core::smart_refctd_ptr(lastSync));
                }
            }

            if (_fence)
                static_cast<COpenGLFence*>(_fence)->associateGLSync(std::move(lastSync));
        }

    private:
        COpenGLFunctionTable* m_gl;
};

}}

#endif