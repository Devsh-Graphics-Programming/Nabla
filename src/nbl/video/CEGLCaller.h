#ifndef __NBL_C_EGL_CALLER_H_INCLUDED__
#define __NBL_C_EGL_CALLER_H_INCLUDED__

#include <EGL/egl.h> // include egl.h from our 3rdparties
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/system/FuncPtrLoader.h"

namespace nbl {
namespace video {
namespace egl
{

namespace impl
{
    class CEGLFuncLoader final : public system::FuncPtrLoader
    {
    public:
        CEGLFuncLoader()
        {
    #ifdef _NBL_USE_SYSTEM_EGL
            lib = dlopen("libEGL.so", RTLD_LAZY);
    #endif
        }

        inline bool isLibraryLoaded() override final
        {
    #ifdef _NBL_USE_SYSTEM_EGL
            return !lib;
    #else
            return true;
    #endif
        }

        inline void* loadFuncPtr(const char* name) override final
        {
    #ifdef _NBL_USE_SYSTEM_EGL
            return dlsym(lib, name);
    #else
            return nullptr;
    #endif
        }

    private:
    #ifdef _NBL_USE_SYSTEM_EGL
        void* lib = nullptr;
    #endif
    };
}

class CEGLCaller final : public system::DynamicFunctionCallerBase<impl::CEGLFuncLoader>
{
    using Base = system::DynamicFunctionCallerBase<impl::CEGLFuncLoader>;

public:
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

#ifdef _NBL_USE_SYSTEM_EGL
#define NBL_IMPL_INIT_EGL_FUNCPTR(FUNC_NAME) NBL_SYSTEM_IMPL_INIT_DYNLIB_FUNCPTR(FUNC_NAME)
#else
#define NBL_IMPL_INIT_EGL_FUNCPTR(FUNC_NAME) ,p ## FUNC_NAME ## (&::FUNC_NAME)
#endif

#define NBL_IMPL_INIT_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_IMPL_INIT_EGL_FUNCPTR,__VA_ARGS__)

#define NBL_IMPL_SWAP_EGL_FUNC_PTRS(...)\
    NBL_FOREACH(NBL_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR,__VA_ARGS__);

    CEGLCaller() :
        Base()
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

private:
    NBL_IMPL_DECLARE_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
};
}
}
}

#endif