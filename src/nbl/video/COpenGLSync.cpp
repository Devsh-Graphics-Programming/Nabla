#include "nbl/video/COpenGLSync.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{
    
COpenGLSync::COpenGLSync(core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& _dev, IOpenGL_FunctionTable* _gl, signalled_t signalled)
    : device(_dev), lockedTable(nullptr), haveNotWaitedOnQueueMask(~(0x1ull<<_gl->getGUID())), cachedRetval(ES_ALREADY_SIGNALED), sync(nullptr)
{
}

COpenGLSync::~COpenGLSync()
{
    if (device && sync)
        device->destroySync(sync);
}

void COpenGLSync::init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& _dev, IOpenGL_FunctionTable* _gl, bool _lockToQueue)
{
    device = _dev;
    sync = _gl->glSync.pglFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (_lockToQueue)
        lockedTable = _gl;
    else
        _gl->glGeneral.pglFlush();
    haveNotWaitedOnQueueMask ^= (1ull << _gl->getGUID());
}

uint64_t COpenGLSync::prewait() const
{
    if (sync)
        return 0ull;

    using clock_t = std::chrono::steady_clock;
    auto start = clock_t::now();
    while (!sync)
    { 
        std::this_thread::yield();
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now() - start).count();
}

COpenGLSync::E_STATUS COpenGLSync::waitCPU(IOpenGL_FunctionTable* _gl, uint64_t timeout)
{
    assert(!lockedTable || lockedTable==_gl);
    if (cachedRetval != ES_TIMEOUT_EXPIRED)
        return cachedRetval;
            
    const uint64_t spintime = prewait();
    if (spintime > timeout)
        return (cachedRetval = ES_TIMEOUT_EXPIRED);
    timeout -= spintime;

    GLenum status = _gl->glSync.pglClientWaitSync(sync, lockedTable ? GL_SYNC_FLUSH_COMMANDS_BIT:0, timeout);
    switch (status)
    {
    case GL_ALREADY_SIGNALED:
        return (cachedRetval = ES_ALREADY_SIGNALED);
        break;
    case GL_TIMEOUT_EXPIRED:
        return (cachedRetval = ES_TIMEOUT_EXPIRED);
        break;
    case GL_CONDITION_SATISFIED:;
        return (cachedRetval = ES_CONDITION_SATISFIED);
        break;
    default:
        break;
    }
    return (cachedRetval = ES_FAIL);
}

void COpenGLSync::waitGPU(IOpenGL_FunctionTable* _gl)
{
    if (!lockedTable) // OpenGL device does not need to wait on itself within the same context
    {
        const uint64_t queueMask = 1ull << _gl->getGUID();
        if (haveNotWaitedOnQueueMask.load() & queueMask)
        {
            prewait();
            _gl->glSync.pglWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
            haveNotWaitedOnQueueMask ^= queueMask;
        }
    }
    else
    {
        assert(lockedTable==_gl);
    }
}

}