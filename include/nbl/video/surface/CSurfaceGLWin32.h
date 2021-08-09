#ifndef __NBL_C_SURFACE_GL_WIN32_H_INCLUDED__
#define __NBL_C_SURFACE_GL_WIN32_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"

#ifdef _NBL_PLATFORM_WINDOWS_

#include "nbl/video/surface/ISurfaceWin32.h"
#include "nbl/video/surface/ISurfaceGL.h"

namespace nbl::video
{

class CSurfaceGLWin32 final : public ISurfaceWin32, public ISurfaceGL
{
    public:
        explicit CSurfaceGLWin32(SCreationParams&& params) : ISurfaceWin32(std::move(params)), ISurfaceGL(params.hwnd)
        {
        }
};

}

#endif

#endif