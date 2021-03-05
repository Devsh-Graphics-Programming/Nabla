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
    COpenGLFence(IOpenGL_LogicalDevice* dev, IOpenGL_FunctionTable* gl) : IGPUFence(dev, ECF_SIGNALED_BIT), IOpenGLSyncPrimitiveBase(dev)
    {
        signal(gl);
    }
    // un-signaled ctor
    explicit COpenGLFence(IOpenGL_LogicalDevice* dev) : IGPUFence(dev, static_cast<E_CREATE_FLAGS>(0)), IOpenGLSyncPrimitiveBase(dev)
    {

    }

    E_STATUS wait(IOpenGL_FunctionTable* _gl, uint64_t timeout)
    {
        const uint64_t pretimeout = prewait();

        // TODO
        // im not sure if this is proper vulkan emulation
        // @matt ?
        {
            if (pretimeout >= timeout)
                return ES_TIMEOUT;
            timeout -= pretimeout;
        }

        COpenGLSync::E_STATUS status = m_sync->waitCPU(_gl, timeout);
        if (status == COpenGLSync::ES_FAIL)
            return ES_ERROR;
        else if (status == COpenGLSync::ES_TIMEOUT_EXPIRED)
            return ES_TIMEOUT;
        else
            return ES_SUCCESS;
    }

    void reset()
    {
        IOpenGLSyncPrimitiveBase::reset();
    }

private:
    IOpenGL_LogicalDevice* m_device;
    core::smart_refctd_ptr<COpenGLSync> m_sync;
};

}}

#endif