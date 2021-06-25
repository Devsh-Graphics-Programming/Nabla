#ifndef __NBL_C_OPENGL_FENCE_H_INCLUDED__
#define __NBL_C_OPENGL_FENCE_H_INCLUDED__

#include "nbl/video/IGPUFence.h"
#include "nbl/video/IOpenGLSyncPrimitiveBase.h"
#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl {
namespace video
{

class IOpenGL_LogicalDevice;

class COpenGLFence final : public IGPUFence, public IOpenGLSyncPrimitiveBase
{
protected:
    ~COpenGLFence() = default;

public:
    // signaled ctor
    COpenGLFence(IOpenGL_LogicalDevice* dev, ILogicalDevice* _dev, IOpenGL_FunctionTable* gl) : IGPUFence(_dev, ECF_SIGNALED_BIT)
    {
        auto sync = core::make_smart_refctd_ptr<COpenGLSync>();
        sync->initSignaled(dev, gl);
        associateGLSync(std::move(sync));
    }
    // un-signaled ctor
    explicit COpenGLFence(ILogicalDevice* _dev) : IGPUFence(_dev, static_cast<E_CREATE_FLAGS>(0))
    {

    }

    E_STATUS wait(IOpenGL_FunctionTable* _gl, uint64_t timeout)
    {
        COpenGLSync::E_STATUS status = m_sync->waitCPU(_gl, timeout);
        if (status == COpenGLSync::ES_FAIL)
            return ES_ERROR;
        else if (status == COpenGLSync::ES_TIMEOUT_EXPIRED)
            return ES_TIMEOUT;
        else
            return ES_SUCCESS;
    }

    E_STATUS getStatus(IOpenGL_FunctionTable* _gl)
    {
        if (!m_sync->isInitialized())
            return ES_NOT_READY;
        auto status = m_sync->waitCPU(_gl, 0ull);
        switch (status)
        {
        case COpenGLSync::ES_TIMEOUT_EXPIRED:
            return ES_NOT_READY;
        case COpenGLSync::ES_FAIL:
            return ES_ERROR;
        default:
            return ES_SUCCESS;
        }
    }

    void reset()
    {
        IOpenGLSyncPrimitiveBase::reset();
    }

private:
    IOpenGL_LogicalDevice* m_device;
};

}}

#endif