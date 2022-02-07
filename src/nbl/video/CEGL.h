#ifndef __NBL_C_EGL_H_INCLUDED__
#define __NBL_C_EGL_H_INCLUDED__

//#include "nbl/video/CWaylandCaller.h"
#include "nbl/video/CEGLCaller.h"

namespace nbl::video::egl
{
class CEGL
{
public:
    CEGL(const char* eglOptionalPath)
        : call(eglOptionalPath) {}

    bool initialize()
    {
        if(display != EGL_NO_DISPLAY)
            return true;

        display = call.peglGetDisplay(EGL_DEFAULT_DISPLAY);
        if(display == EGL_NO_DISPLAY)
            return false;
        if(!call.peglInitialize(display, &version.major, &version.minor))
            return false;
        return true;
    }
    bool deinitialize()
    {
        if(display != EGL_NO_DISPLAY)
            return call.peglTerminate(display);
        return true;
    }

    CEGLCaller call;
    EGLDisplay display = EGL_NO_DISPLAY;
    struct
    {
        EGLint major;
        EGLint minor;
    } version;
    /*
    #ifdef _NBL_BUILD_WITH_WAYLAND
        CWaylandCaller wlcall;
        struct wl_display* wldisplay = NULL;
    #endif
        */
};

}

#endif
