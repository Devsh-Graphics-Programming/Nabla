#ifndef __NBL_C_OPENGL_SWAPCHAIN_H_INCLUDED__
#define __NBL_C_OPENGL_SWAPCHAIN_H_INCLUDED__

#include "nbl/video/COpenGL_Swapchain.h"
#include "nbl/video/COpenGLFunctionTable.h"

namespace nbl {
namespace video
{

using COpenGLSwapchain = COpenGL_Swapchain<COpenGLFunctionTable>;

}
}


#endif