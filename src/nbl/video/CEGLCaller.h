#ifndef __NBL_C_EGL_CALLER_H_INCLUDED__
#define __NBL_C_EGL_CALLER_H_INCLUDED__

#include "EGL/egl.h"
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/system/DefaultFuncPtrLoader.h"

namespace nbl::video::egl
{

class CEGLCaller final
{
#define NBL_EGL_FUNC_LIST \
    eglChooseConfig,\
    eglCopyBuffers,\
    eglCreateContext,\
    eglCreatePbufferSurface,\
    eglCreatePixmapSurface,\
    eglCreateWindowSurface,\
    eglDestroyContext,\
    eglDestroySurface,\
    eglGetConfigAttrib,\
    eglGetConfigs,\
    eglGetCurrentDisplay,\
    eglGetCurrentSurface,\
    eglGetDisplay,\
    eglGetError,\
    eglGetProcAddress,\
    eglInitialize,\
    eglMakeCurrent,\
    eglQueryContext,\
    eglQueryString,\
    eglQuerySurface,\
    eglSwapBuffers,\
    eglTerminate,\
    eglWaitGL,\
    eglWaitNative,\
    \
    eglBindTexImage,\
    eglReleaseTexImage,\
    eglSurfaceAttrib,\
    eglSwapInterval,\
    \
    eglBindAPI,\
    eglQueryAPI,\
    eglCreatePbufferFromClientBuffer,\
    eglReleaseThread,\
    eglWaitClient,\
    \
    eglGetCurrentContext,\
    \
    eglCreateSync,\
    eglDestroySync,\
    eglClientWaitSync,\
    eglGetSyncAttrib,\
    eglCreateImage,\
    eglDestroyImage,\
    eglGetPlatformDisplay,\
    eglCreatePlatformWindowSurface,\
    eglCreatePlatformPixmapSurface,\
    eglWaitSync,\
    eglGetPlatformDependentHandles

#define NBL_IMPL_DECLARE_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR,__VA_ARGS__);

#define NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) reinterpret_cast<void*>(&::FUNC_NAME)

#define NBL_IMPL_INIT_EGL_FUNCPTR(FUNC_NAME) ,p ## FUNC_NAME ( NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) )

#define NBL_IMPL_INIT_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_IMPL_INIT_EGL_FUNCPTR,__VA_ARGS__)

#define NBL_IMPL_SWAP_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR,__VA_ARGS__);

public:
    CEGLCaller() : dummy(0)
        NBL_IMPL_INIT_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
#if !defined(_NBL_PLATFORM_ANDROID_)
        NBL_IMPL_INIT_EGL_FUNCPTR(eglGetPlatformDependentHandles)
#endif
    {
    }

    CEGLCaller(CEGLCaller&& other)
    {
        operator=(std::move(other));
    }

    CEGLCaller& operator=(CEGLCaller&& other)
    {
        NBL_IMPL_SWAP_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST);
#if !defined(_NBL_PLATFORM_ANDROID_)
        std::swap(peglGetPlatformDependentHandles, other.peglGetPlatformDependentHandles);
#endif

        return *this;
    }

    int dummy;
    NBL_IMPL_DECLARE_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
#if !defined(_NBL_PLATFORM_ANDROID_)
    NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR(eglGetPlatformDependentHandles);
#endif
};

}

#endif