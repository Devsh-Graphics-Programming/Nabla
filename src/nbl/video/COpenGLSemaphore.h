#ifndef __NBL_C_OPENGL_SEMAPHORE_H_INCLUDED__
#define __NBL_C_OPENGL_SEMAPHORE_H_INCLUDED__

#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/COpenGLSync.h"
#include "COpenGLExtensionHandler.h"

namespace nbl {
namespace video
{

class COpenGLSemaphore : public IGPUSemaphore
{
protected:
    ~COpenGLSemaphore()
    {
#ifdef _NBL_DEBUG
        if (m_sync)
        {
            auto status = m_sync->waitCPU(0);
            assert(status == COpenGLSync::ES_CONDITION_SATISFIED || status == COpenGLSync::ES_ALREADY_SIGNALED);
        }
#endif
    }

public:
    enum E_STATUS
    {
        ES_SUCCESS,
        ES_TIMEOUT,
        ES_NOT_READY,
        ES_ERROR
    };

    COpenGLSemaphore()
    {

    }

    inline COpenGLSync* getInternalObject() { return m_sync.get(); }

    void wait()
    {
        m_sync->waitGPU();
    }

    void signal(core::smart_refctd_ptr<COpenGLSync>&& _sync)
    {
        m_sync = std::move(_sync);
        glFlush();
    }

private:
    bool isWaitable() const { return static_cast<bool>(m_sync); }
    bool isSignalable() const { return !m_sync; }

    core::smart_refctd_ptr<COpenGLSync> m_sync = nullptr;
};

}}

#endif