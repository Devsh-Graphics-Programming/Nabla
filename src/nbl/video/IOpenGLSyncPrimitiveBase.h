#ifndef __NBL_I_OPENGL_SYNC_PRIMITIVE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_SYNC_PRIMITIVE_BASE_H_INCLUDED__

#include "nbl/core/compile_config.h"
#include "nbl/video/COpenGLSync.h"
#include <atomic>
#include <chrono>

namespace nbl {
namespace video
{

class IOpenGLSyncPrimitiveBase
{
protected:
    IOpenGLSyncPrimitiveBase() = default;

public:
    virtual ~IOpenGLSyncPrimitiveBase() = default;

    inline void associateGLSync(core::smart_refctd_ptr<COpenGLSync>&& sync)
    {
        //assert(!m_sync);
        //semaphores can be reused, we just need to guarantee that a semaphore is unsignaled when signal operation happens on GPU (https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit.html)
        m_sync = std::move(sync);
    }

    inline COpenGLSync* getInternalObject() { return m_sync.get(); }

protected:
    // reset to unsignaled state
    inline void reset()
    {
        m_sync = nullptr;
    }

    core::smart_refctd_ptr<COpenGLSync> m_sync;
};

}
}

#endif
