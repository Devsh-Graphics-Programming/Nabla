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

        inline COpenGLSync(IOpenGL_LogicalDevice* _dev, IOpenGL_FunctionTable* _gl) : device(_dev), cachedRetval(ES_TIMEOUT_EXPIRED)
        {
            sync = _gl->glSync.pglFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }

        E_STATUS waitCPU(IOpenGL_FunctionTable* _gl, uint64_t timeout)
        {
            if (cachedRetval != ES_TIMEOUT_EXPIRED)
                return cachedRetval;

            GLenum status = _gl->glSync.pglClientWaitSync(sync, 0, timeout);
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
            _gl->glSync.pglWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
        }

        inline GLsync getOpenGLName() const
        {
            return sync;
        }

    private:
        IOpenGL_LogicalDevice* device;
        E_STATUS cachedRetval;
        GLsync sync;
};

}}

#endif