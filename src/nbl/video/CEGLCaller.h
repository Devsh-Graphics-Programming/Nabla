#ifndef __NBL_C_EGL_CALLER_H_INCLUDED__
#define __NBL_C_EGL_CALLER_H_INCLUDED__

#include <EGL/egl.h> // include egl.h from our 3rdparties
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/system/DefaultFuncPtrLoader.h"

namespace nbl {
namespace video {
namespace egl
{

namespace impl
{
    using CEGLFuncLoader = system::DefaultFuncPtrLoader;
}

class CEGLCaller final : public system::DynamicFunctionCallerBase<impl::CEGLFuncLoader>
{
    using Base = system::DynamicFunctionCallerBase<impl::CEGLFuncLoader>;

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
    eglWaitSync

#define NBL_IMPL_DECLARE_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR,__VA_ARGS__);

#ifdef _NBL_PLATFORM_WINDOWS_
#define NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) &::FUNC_NAME
#else
#define NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) [this] () -> void* { \
    if (auto fptr = Base::loader.loadFuncPtr( #FUNC_NAME ))\
        return fptr;\
    return reinterpret_cast<void*>(&::FUNC_NAME);\
}()
#endif

#define _INDIRECTION1(X) (X)
#define NBL_IMPL_INIT_EGL_FUNCPTR(FUNC_NAME) ,p ## FUNC_NAME ( NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) )

#define NBL_IMPL_INIT_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_IMPL_INIT_EGL_FUNCPTR,__VA_ARGS__)

#define NBL_IMPL_SWAP_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR,__VA_ARGS__);

    constexpr inline static const char* LibName = "libEGL";

public:
    CEGLCaller() :
        Base(LibName)
        NBL_IMPL_INIT_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
    {

    }

    CEGLCaller(CEGLCaller&& other)
    {
        operator=(std::move(other));
    }

    CEGLCaller& operator=(CEGLCaller&& other)
    {
        Base::operator=(std::move(other));
        NBL_IMPL_SWAP_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST);

        return *this;
    }

    NBL_IMPL_DECLARE_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
};

}
}
}

#endif