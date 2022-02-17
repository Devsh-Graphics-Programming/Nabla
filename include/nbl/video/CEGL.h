#ifndef _NBL_C_EGL_H_INCLUDED_
#define _NBL_C_EGL_H_INCLUDED_

//#include "nbl/video/CWaylandCaller.h"
#include "nbl/video/CEGLCaller.h"

namespace nbl::video::egl
{

// any EGL handles are not native EGL handles, they come from our EGL over WGL/GLX/EGL implementation!
class CEGL
{
    public:
        //
        struct Context
        {
            EGLContext ctx = EGL_NO_CONTEXT;
            EGLSurface surface = EGL_NO_SURFACE;

            // to load function pointers, make EGL context current and use `egl->call.peglGetProcAddress("glFuncname")`
        };

        //
        CEGL(const char* eglOptionalPath) : call(eglOptionalPath) {}

        bool initialize()
        {
            if (display != EGL_NO_DISPLAY)
                return true;

            display = call.peglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (display == EGL_NO_DISPLAY)
                return false;
            if (!call.peglInitialize(display, &version.major, &version.minor))
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
        struct {
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
