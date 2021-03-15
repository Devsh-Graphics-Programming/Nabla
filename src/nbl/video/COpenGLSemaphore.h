#ifndef __NBL_C_OPENGL_SEMAPHORE_H_INCLUDED__
#define __NBL_C_OPENGL_SEMAPHORE_H_INCLUDED__

#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/IOpenGLSyncPrimitiveBase.h"

namespace nbl {
namespace video
{

class IOpenGL_LogicalDevice;

class COpenGLSemaphore : public IGPUSemaphore, public IOpenGLSyncPrimitiveBase
{
protected:
    ~COpenGLSemaphore()
    {
/*#ifdef _NBL_DEBUG
        if (m_sync)
        {
            auto status = m_sync->waitCPU(0);
            assert(status == COpenGLSync::ES_CONDITION_SATISFIED || status == COpenGLSync::ES_ALREADY_SIGNALED);
        }
#endif*/
    }

public:
    enum E_STATUS
    {
        ES_SUCCESS,
        ES_TIMEOUT,
        ES_NOT_READY,
        ES_ERROR
    };

    COpenGLSemaphore(IOpenGL_LogicalDevice* dev) : IGPUSemaphore(dev), IOpenGLSyncPrimitiveBase(dev)
    {

    }

    void wait(IOpenGL_FunctionTable* _gl)
    {
        prewait();
        m_sync->waitGPU(_gl);
    }
};

}}

#endif