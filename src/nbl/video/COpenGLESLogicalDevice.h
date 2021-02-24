#ifndef __NBL_C_OPENGLES_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGLES_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/COpenGL_LogicalDevice.h"
#include "nbl/video/COpenGLESQueue.h"
#include "nbl/video/COpenGLESSwapchain.h"

namespace nbl {
namespace video
{

using COpenGLESLogicalDevice = COpenGL_LogicalDevice<COpenGLESQueue, COpenGLESSwapchain>;

}
}

#endif
