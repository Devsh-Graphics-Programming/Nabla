#ifndef __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/COpenGL_LogicalDevice.h"
#include "nbl/video/COpenGLQueue.h"

namespace nbl {
namespace video
{

using COpenGLLogicalDevice = COpenGL_LogicalDevice<COpenGLQueue>;

}
}

#endif
