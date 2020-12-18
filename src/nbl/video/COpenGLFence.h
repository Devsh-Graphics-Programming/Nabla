#ifndef __NBL_C_OPENGL_FENCE_H_INCLUDED__
#define __NBL_C_OPENGL_FENCE_H_INCLUDED__

#include "nbl/video/IGPUFence.h"
#include "nbl/video/COpenGLSync.h"
#include "COpenGLExtensionHandler.h"

namespace nbl {
namespace video
{

class COpenGLFence : public IGPUFence
{
protected:
    ~COpenGLFence()
    {
    }

public:
    COpenGLFence(E_CREATE_FLAGS flags) : IGPUFence(flags), m_sync()
    {
    }

    void associateGLSync(core::smart_refctd_ptr<COpenGLSync>&& _sync)
    {
        setStatus(ES_NOT_READY); // or TIMEOUT ?
        m_sync = std::move(_sync);
    }
/*
    E_STATUS wait(uint64_t timeout) override
    {
        if (!m_sync)
            return setStatus(ES_SUCCESS);

        auto status = m_sync->waitCPU(timeout);
        switch (status)
        {
        case COpenGLSync::ES_ALREADY_SIGNALED: [[fallthrough]];
        case COpenGLSync::ES_CONDITION_SATISFIED:
            return setStatus(ES_SUCCESS);
        case COpenGLSync::ES_TIMEOUT_EXPIRED:
            return setStatus(ES_TIMEOUT);
        case COpenGLSync::ES_FAIL: [[fallthrough]];
        default:
            return setStatus(ES_ERROR);
        }
    }

    void reset() override
    {
        assert(wait() == SUCCESS);
        m_sync = nullptr;
        m_status = ES_NOT_READY;
    }

    E_STATUS query() const override
    {
        return const_cast<COpenGLFence*>(this)->wait(0ull);
    }
*/
private:
    core::smart_refctd_ptr<COpenGLSync> m_sync;
};

}}

#endif