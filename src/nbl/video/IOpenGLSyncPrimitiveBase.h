#ifndef __NBL_I_OPENGL_SYNC_PRIMITIVE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_SYNC_PRIMITIVE_BASE_H_INCLUDED__

#include "nbl/core/compile_config.h"
#include "nbl/video/COpenGLSync.h"
#include <atomic>
#include <chrono>

namespace nbl {
namespace video
{

class IOpenGL_LogicalDevice;

class IOpenGLSyncPrimitiveBase
{
protected:
    explicit IOpenGLSyncPrimitiveBase(IOpenGL_LogicalDevice* dev) : m_device(dev), m_toBeSignaled(false) {}

public:
    virtual ~IOpenGLSyncPrimitiveBase() = default;

    inline void setToBeSignaled() 
    { 
        if (!m_sync)
            m_toBeSignaled = true;
    }
    // Answer to "what if the sync is waited upon before it was signaled (before actual GLsync exist) ?"
    inline uint64_t prewait() const
    {
        if (!m_toBeSignaled.load())
            return 0ull;

        using clock_t = std::chrono::high_resolution_clock;
        auto start = clock_t::now();
        while (m_toBeSignaled.load()) { ; }

        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now() - start).count();
    }
    inline void signal(IOpenGL_FunctionTable* _gl)
    {
        m_sync = core::make_smart_refctd_ptr<COpenGLSync>(m_device, _gl);
        m_toBeSignaled = false;
    }
    inline void signal(core::smart_refctd_ptr<COpenGLSync>&& sync)
    {
        m_sync = std::move(sync);
        m_toBeSignaled = false;
    }

    inline bool isWaitable() const { return static_cast<bool>(m_sync) || m_toBeSignaled.load(); }
    inline bool isSignalable() const { return !m_sync; }

    inline COpenGLSync* getInternalObject() { return m_sync.get(); }

protected:
    // reset to unsignaled state
    inline void reset()
    {
        m_toBeSignaled.store(false);
        m_sync = nullptr;
    }

    IOpenGL_LogicalDevice* m_device;
    core::smart_refctd_ptr<COpenGLSync> m_sync;

private:
    std::atomic_bool m_toBeSignaled;
};

}
}

#endif
