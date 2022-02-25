#ifndef __NBL_C_OPENGL_SYNC_H_INCLUDED__
#define __NBL_C_OPENGL_SYNC_H_INCLUDED__

#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace video
{

class IOpenGL_LogicalDevice;

class COpenGLSync final : public core::IReferenceCounted
{
    protected:
        virtual ~COpenGLSync();

    public:
        enum E_STATUS : uint32_t
        {
            //! Indicates that an error occurred. Additionally, an OpenGL error will be generated.
            ES_FAIL = 0,
            //! If it returns GL_TIMEOUT_EXPIRED, then the sync object did not signal within the given timeout period (includes before us calling the func).
            ES_TIMEOUT_EXPIRED,
            //! Indicates that sync​ was signaled before the timeout expired.
            ES_CONDITION_SATISFIED,
            //! GPU already completed work before we even asked == THIS IS WHAT WE WANT
            ES_ALREADY_SIGNALED
        };

        inline COpenGLSync() : device(nullptr), lockedTable(nullptr), haveNotWaitedOnQueueMask(~0ull), cachedRetval(ES_TIMEOUT_EXPIRED), sync(nullptr)
        {

        }

        void init(IOpenGL_LogicalDevice* _dev, IOpenGL_FunctionTable* _gl, bool _lockToQueue = false)
        {
            device = _dev;
            sync = _gl->glSync.pglFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            if (_lockToQueue)
                lockedTable = _gl;
            else
                _gl->glGeneral.pglFlush();
            haveNotWaitedOnQueueMask ^= (1ull << _gl->getGUID());
        }
        // for fences
        void initSignaled(IOpenGL_LogicalDevice* _dev, IOpenGL_FunctionTable* _gl)
        {
            device = _dev;
            haveNotWaitedOnQueueMask ^= (1ull << _gl->getGUID());
            cachedRetval = ES_ALREADY_SIGNALED;
        }

        uint64_t prewait() const
        {
            if (sync)
                return 0ull;

            using clock_t = std::chrono::high_resolution_clock;
            auto start = clock_t::now();
            while (!sync)
            { 
                std::this_thread::yield();
            }

            return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now() - start).count();
        }

        E_STATUS waitCPU(IOpenGL_FunctionTable* _gl, uint64_t timeout)
        {
            assert(!lockedTable || lockedTable==_gl);
            if (cachedRetval != ES_TIMEOUT_EXPIRED)
                return cachedRetval;
            
            const uint64_t spintime = prewait();
            if (spintime > timeout)
                return (cachedRetval = ES_TIMEOUT_EXPIRED);
            timeout -= spintime;

            GLenum status = _gl->glSync.pglClientWaitSync(sync, lockedTable?GL_SYNC_FLUSH_COMMANDS_BIT:0, timeout); // GL_SYNC_FLUSH_COMMANDS_BIT to flags?
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

        void waitGPU(IOpenGL_FunctionTable* _gl)
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

        inline GLsync getOpenGLName() const
        {
            return sync;
        }

        bool isInitialized() const
        {
            // (!sync && (cachedRetval == ES_ALREADY_SIGNALED)) means initialized as signaled
            return static_cast<bool>(sync) || cachedRetval == ES_ALREADY_SIGNALED;
        }

    private:
        IOpenGL_LogicalDevice* device;
        IOpenGL_FunctionTable* lockedTable;
        std::atomic_uint64_t haveNotWaitedOnQueueMask;
        E_STATUS cachedRetval;
        volatile GLsync sync;
};

}}

#endif