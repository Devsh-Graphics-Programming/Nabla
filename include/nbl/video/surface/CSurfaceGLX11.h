#ifndef __NBL_C_SURFACE_GL_LINUX_H_INCLUDED__
#define __NBL_C_SURFACE_GL_LINUX_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/video/surface/ISurfaceX11.h"
#include "nbl/video/surface/ISurfaceGL.h"

namespace nbl {
namespace video
{

class CSurfaceGLX11 final : public ISurfaceX11, public ISurfaceGL
{
public:
    explicit CSurfaceGLX11(SCreationParams&& params) : ISurfaceX11(std::move(params)), ISurfaceGL(params.window)
    {

    }
};

}
}

#endif

#endif