#ifndef __NBL_C_SURFACE_GL_WIN32_H_INCLUDED__
#define __NBL_C_SURFACE_GL_WIN32_H_INCLUDED__

#include "nbl/video/surface/ISurfaceWin32.h"
#include "nbl/video/surface/ISurfaceGL.h"

namespace nbl {
namespace video
{

class CSurfaceGLWin32 final : public ISurfaceWin32, public ISurfaceGL
{
public:
    CSurfaceGLWin32(SCreationParams&& params) : ISurfaceWin32(std::move(params)), ISurfaceGL(params.hwnd)
    {

    }
};

}
}

#endif