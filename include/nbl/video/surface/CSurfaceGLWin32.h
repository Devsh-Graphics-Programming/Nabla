#ifndef __NBL_C_SURFACE_GL_WIN32_H_INCLUDED__
#define __NBL_C_SURFACE_GL_WIN32_H_INCLUDED__

#include "nbl/video/surface/ISurfaceWin32.h"

namespace nbl {
namespace video
{

class CSurfaceGLWin32 : public ISurfaceWin32
{
    // no specific object like VkSurfaceKHR here,
    // it just needs to have a GL context assigned

};

}
}

#endif