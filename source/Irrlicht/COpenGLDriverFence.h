// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

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
    public:
        inline COpenGLDriverFence() : cachedRetval(EDFR_TIMEOUT_EXPIRED)
        {
            fence = COpenGLExtensionHandler::extGlFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
            firstTimeFlush = !COpenGLExtensionHandler::IsIntelGPU;
            if (COpenGLExtensionHandler::IsIntelGPU)
                glFlush();
        }

        virtual ~COpenGLDriverFence()
        {
            COpenGLExtensionHandler::extGlDeleteSync(fence);
        }

        virtual E_DRIVER_FENCE_RETVAL waitCPU(const uint64_t &timeout, const bool &flush=false)
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

        virtual void waitGPU()
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
