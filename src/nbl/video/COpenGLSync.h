#ifndef __NBL_C_OPENGL_SYNC_H_INCLUDED__
#define __NBL_C_OPENGL_SYNC_H_INCLUDED__

#include "COpenGLExtensionHandler.h"

namespace nbl {
namespace video
{

class COpenGLSync
{
    protected:
        virtual ~COpenGLSync()
        {
            COpenGLExtensionHandler::extGlDeleteSync(fence);
        }

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

        inline COpenGLSync() : cachedRetval(ES_TIMEOUT_EXPIRED)
        {
            fence = COpenGLExtensionHandler::extGlFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }

        E_STATUS waitCPU(const uint64_t& timeout)
        {
            if (cachedRetval != ES_TIMEOUT_EXPIRED)
                return cachedRetval;

            GLenum status = COpenGLExtensionHandler::extGlClientWaitSync(fence, 0, timeout);
            switch (status)
            {
            case GL_ALREADY_SIGNALED:
                return cachedRetval = ES_ALREADY_SIGNALED;
                break;
            case GL_TIMEOUT_EXPIRED:
                return cachedRetval = ES_TIMEOUT_EXPIRED;
                break;
            case GL_CONDITION_SATISFIED:;
                return cachedRetval = ES_ALREADY_SIGNALED;
                break;
            default:
                break;
            }
            return cachedRetval = ES_FAIL;
        }

        void waitGPU()
        {
            COpenGLExtensionHandler::extGlWaitSync(fence, 0, GL_TIMEOUT_IGNORED);
        }

    private:
        E_STATUS cachedRetval;
        GLsync fence;
};

}}

#endif