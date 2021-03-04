#ifndef __NBL_C_OPENGL_CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL_CONNECTION_H_INCLUDED__

#include "nbl/video/COpenGL_Connection.h"
#include "nbl/video/COpenGLESPhysicalDevice.h"

namespace nbl {
namespace video
{

using COpenGLESConnection = COpenGL_Connection<COpenGLESPhysicalDevice, EAT_OPENGL_ES>;

}
}

#endif
