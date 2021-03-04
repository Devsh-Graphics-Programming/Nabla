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
    bool initialize()
    {
        if (display != EGL_NO_DISPLAY)
            return true;

        display = call.peglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY)
            return false;
        if (!call.peglInitialize(display, nullptr, nullptr))
            return false;
        return true;
    }
    bool deinitialize()
    {
        if (display != EGL_NO_DISPLAY)
            return call.peglTerminate(display);
        return true;
    }

    CEGLCaller call;
    EGLDisplay display = EGL_NO_DISPLAY;
};

}
}
}

#endif
