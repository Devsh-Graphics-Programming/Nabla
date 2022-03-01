#ifndef _NBL_C_OPENGL_SYNC_H_INCLUDED_
#define _NBL_C_OPENGL_SYNC_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

namespace nbl::video
{

class IOpenGL_LogicalDevice;
class IOpenGL_FunctionTable;

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

        // External GLsync Import constructor. If you leave the `_dev` parameter null, then you're in charge of deleting the `_sync` by yourself.
        inline COpenGLSync(GLsync _sync, core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& _dev=nullptr)
            : device(std::move(_dev)), lockedTable(nullptr), haveNotWaitedOnQueueMask(~0ull), cachedRetval(ES_TIMEOUT_EXPIRED), sync(_sync) {}

        //
        inline GLsync getOpenGLName() const {return sync;}

        // Whether Nabla flushed after placing a fence or not, counting on an implicit flush from waiting on the same context.
        // You can still wait on the fence with a different context, but it the context it was placed on needs a flush first.
        inline bool initializedWithImplicitFlush() const {return lockedTable;}

        // whether `init` has been called and there is a valid `sync` to wait for
        bool isInitialized() const
        {
            // (!sync && (cachedRetval == ES_ALREADY_SIGNALED)) means initialized as signaled
            return static_cast<bool>(sync) || cachedRetval == ES_ALREADY_SIGNALED;
        }


        // for internal Nabla usage
        void init(core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& _dev, IOpenGL_FunctionTable* _gl, bool _lockToQueue = false);
        uint64_t prewait() const;
        E_STATUS waitCPU(IOpenGL_FunctionTable* _gl, uint64_t timeout);
        void waitGPU(IOpenGL_FunctionTable* _gl);
        
        // Empty (waiting to be made pending a signal) not yet initialized sync constructor
        inline COpenGLSync() : COpenGLSync(nullptr,nullptr) {}
        // Already signalled constructor (for IGPUFence)
        struct signalled_t {};
        constexpr static inline signalled_t signalled = {};
        COpenGLSync(core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& _dev, IOpenGL_FunctionTable* _gl, signalled_t signalled);

    private:
        core::smart_refctd_ptr<IOpenGL_LogicalDevice> device;
        IOpenGL_FunctionTable* lockedTable;
        std::atomic_uint64_t haveNotWaitedOnQueueMask;
        E_STATUS cachedRetval;
        // maybe atomic instead of volatile would work better?
        volatile GLsync sync;
};

}

#endif