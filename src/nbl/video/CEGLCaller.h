#ifndef __NBL_C_EGL_CALLER_H_INCLUDED__
#define __NBL_C_EGL_CALLER_H_INCLUDED__

#include "EGL/egl.h"
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/system/DefaultFuncPtrLoader.h"

namespace nbl::video::egl
{
#define NBL_EGL_FUNC_LIST                 \
    eglChooseConfig,                      \
        eglCopyBuffers,                   \
        eglCreateContext,                 \
        eglCreatePbufferSurface,          \
        eglCreatePixmapSurface,           \
        eglCreateWindowSurface,           \
        eglDestroyContext,                \
        eglDestroySurface,                \
        eglGetConfigAttrib,               \
        eglGetConfigs,                    \
        eglGetCurrentDisplay,             \
        eglGetCurrentSurface,             \
        eglGetDisplay,                    \
        eglGetError,                      \
        eglGetProcAddress,                \
        eglInitialize,                    \
        eglMakeCurrent,                   \
        eglQueryContext,                  \
        eglQueryString,                   \
        eglQuerySurface,                  \
        eglSwapBuffers,                   \
        eglTerminate,                     \
        eglWaitGL,                        \
        eglWaitNative,                    \
                                          \
        eglBindTexImage,                  \
        eglReleaseTexImage,               \
        eglSurfaceAttrib,                 \
        eglSwapInterval,                  \
                                          \
        eglBindAPI,                       \
        eglQueryAPI,                      \
        eglCreatePbufferFromClientBuffer, \
        eglReleaseThread,                 \
        eglWaitClient,                    \
                                          \
        eglGetCurrentContext,             \
                                          \
        eglCreateSync,                    \
        eglDestroySync,                   \
        eglClientWaitSync,                \
        eglGetSyncAttrib,                 \
        eglCreateImage,                   \
        eglDestroyImage,                  \
        eglGetPlatformDisplay,            \
        eglCreatePlatformWindowSurface,   \
        eglCreatePlatformPixmapSurface,   \
        eglWaitSync,                      \
        eglGetPlatformDependentHandles

#define NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) reinterpret_cast<void*>(&::FUNC_NAME)
class CEGLLoader : public system::FuncPtrLoader
{
    system::DefaultFuncPtrLoader m_libEGL;

public:
    CEGLLoader()
        : m_libEGL() {}
    CEGLLoader(const char* eglOptionalPath)
        : m_libEGL(eglOptionalPath) {}
    CEGLLoader(CEGLLoader&& other)
        : m_libEGL()
    {
        operator=(std::move(other));
    }
    ~CEGLLoader() {}

    CEGLLoader& operator=(CEGLLoader&& other)
    {
        m_libEGL = std::move(other.m_libEGL);
        return *this;
    }

    inline bool isLibraryLoaded() override { return true; }

    void* loadFuncPtr(const char* funcname) override
    {
        if(m_libEGL.isLibraryLoaded())
            return m_libEGL.loadFuncPtr(funcname);
#define LOAD_DYNLIB_FUNCPTR(FUNC_NAME)    \
    if(strcmp(funcname, #FUNC_NAME) == 0) \
        return NBL_IMPL_GET_FUNC_PTR(FUNC_NAME);
        NBL_FOREACH(LOAD_DYNLIB_FUNCPTR, NBL_EGL_FUNC_LIST)
#undef LOAD_DYNLIB_FUNCPTR
        return nullptr;
    }
};
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CEGLCaller, CEGLLoader, NBL_EGL_FUNC_LIST);

/*
class CEGLCaller final : public core::Uncopyable
{




    public:
        #define NBL_IMPL_INIT_EGL_FUNCPTR(FUNC_NAME) ,p ## FUNC_NAME ( NBL_IMPL_GET_FUNC_PTR(FUNC_NAME) )
        #define NBL_IMPL_INIT_EGL_FUNC_PTRS(...)\
            NBL_FOREACH(NBL_IMPL_INIT_EGL_FUNCPTR,__VA_ARGS__)
        CEGLCaller() : core::Uncopyable()
            NBL_IMPL_INIT_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
        {
        }
        #undef NBL_IMPL_INIT_EGL_FUNC_PTRS
        #undef NBL_IMPL_INIT_EGL_FUNCPTR
        #undef NBL_IMPL_GET_FUNC_PTR
        CEGLCaller(CEGLCaller&& other)
        {
            operator=(std::move(other));
        }

        CEGLCaller& operator=(CEGLCaller&& other)
        {
            #define NBL_IMPL_SWAP_EGL_FUNC_PTRS(...)\
                NBL_FOREACH(NBL_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR,__VA_ARGS__);
            NBL_IMPL_SWAP_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST);
            #undef NBL_IMPL_SWAP_EGL_FUNC_PTRS

            return *this;
        }

        #define NBL_IMPL_DECLARE_EGL_FUNC_PTRS(...)\
            NBL_FOREACH(NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR,__VA_ARGS__);
        NBL_IMPL_DECLARE_EGL_FUNC_PTRS(NBL_EGL_FUNC_LIST)
        #undef NBL_IMPL_DECLARE_EGL_FUNC_PTRS

};
*/

#undef NBL_EGL_FUNC_LIST
}

#endif