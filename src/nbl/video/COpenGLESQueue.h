#ifndef __NBL_C_OPENGLES_QUEUE_H_INCLUDED__
#define __NBL_C_OPENGLES_QUEUE_H_INCLUDED__

#include "nbl/video/COpenGL_Queue.h"
#include "nbl/video/COpenGLESFunctionTable.h"

namespace nbl {
namespace video
{

using COpenGLESQueue = COpenGL_Queue<COpenGLESFunctionTable>;

}
}

#endif
