#ifndef __NBL_C_SURFACE_GL_WAYLAND_H_INCLUDED__
#define __NBL_C_SURFACE_GL_WAYLAND_H_INCLUDED__

#include "nbl/core/compile_config.h"

#ifdef _NBL_BUILD_WITH_WAYLAND

#include "nbl/video/surface/ISurfaceWayland.h"
#include "nbl/video/surface/ISurfaceGL.h"

namespace nbl {
namespace video
{

class CSurfaceGLWayland final : public ISurfaceWayland, public ISurfaceGL
{
public:
    explicit CSurfaceGLWayland(SCreationParams&& params) : ISurfaceWayland(std::move(params)), ISurfaceGL(params.window)
    {

    }
};

}
}

#endif

#endif