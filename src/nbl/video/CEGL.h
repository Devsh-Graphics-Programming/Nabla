#ifndef __NBL_C_EGL_H_INCLUDED__
#define __NBL_C_EGL_H_INCLUDED__

#include "nbl/video/CEGLCaller.h"

namespace nbl {
namespace video {
namespace egl
{

class CEGL
{
public:
    void initialize()
    {
        if (display != EGL_NO_DISPLAY)
            return;

        display = call.peglGetDisplay(EGL_DEFAULT_DISPLAY);
        call.peglInitialize(display, nullptr, nullptr);
    }
    void deinitialize()
    {
        if (display != EGL_NO_DISPLAY)
            call.peglTerminate(display);
    }

    CEGLCaller call;
    EGLDisplay display = EGL_NO_DISPLAY;
};

}
}
}

#endif
