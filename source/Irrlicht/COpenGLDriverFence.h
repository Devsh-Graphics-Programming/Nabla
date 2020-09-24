// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_OPENGL_DRIVER_FENCE_H_INCLUDED__
#define __C_OPENGL_DRIVER_FENCE_H_INCLUDED__

#include "IDriverFence.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
namespace irr
{
namespace video
{

class COpenGLDriverFence : public IDriverFence
{
    protected:
        virtual ~COpenGLDriverFence()
        {
            COpenGLExtensionHandler::extGlDeleteSync(fence);
        }

    public:
        inline COpenGLDriverFence(const bool& implicitFlushOnCPUWait=false) : cachedRetval(EDFR_TIMEOUT_EXPIRED)
        {
            fence = COpenGLExtensionHandler::extGlFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
            firstTimeFlush = implicitFlushOnCPUWait;
        }

        virtual bool canDeferredFlush() const override {return firstTimeFlush;}

        virtual E_DRIVER_FENCE_RETVAL waitCPU(const uint64_t &timeout, const bool &flush=false) override
        {
            if (cachedRetval!=EDFR_TIMEOUT_EXPIRED)
                return cachedRetval;

            switch(COpenGLExtensionHandler::extGlClientWaitSync(fence,flush&&firstTimeFlush ? GL_SYNC_FLUSH_COMMANDS_BIT:0,timeout))
            {
                case GL_ALREADY_SIGNALED:
                    firstTimeFlush = false;
                    cachedRetval = EDFR_ALREADY_SIGNALED;
                    return EDFR_ALREADY_SIGNALED;
                    break;
                case GL_TIMEOUT_EXPIRED:
                    firstTimeFlush = false;
                    return EDFR_TIMEOUT_EXPIRED;
                    break;
                case GL_CONDITION_SATISFIED:
                    firstTimeFlush = false;
                    cachedRetval = EDFR_ALREADY_SIGNALED;
                    return EDFR_CONDITION_SATISFIED;
                    break;
                default:
                //case GL_WAIT_FAILED:
                    firstTimeFlush = false;
                    cachedRetval = EDFR_FAIL;
                    return EDFR_FAIL;
                    break;
            }
        }

        virtual void waitGPU() override
        {
            COpenGLExtensionHandler::extGlWaitSync(fence,0,GL_TIMEOUT_IGNORED);
            firstTimeFlush = false;
        }
    private:
        E_DRIVER_FENCE_RETVAL cachedRetval;
        bool firstTimeFlush;
        GLsync fence;
};

} // end namespace scene
} // end namespace irr
#endif // _IRR_COMPILE_WITH_OPENGL_

#endif
