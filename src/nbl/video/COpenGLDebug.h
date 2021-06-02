#ifndef __NBL_C_OPENGL_DEBUG_H_INCLUDED__
#define __NBL_C_OPENGL_DEBUG_H_INCLUDED__

#include "nbl/core/compile_config.h"
#include "nbl/video/debug/debug.h"
#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl::video
{

void opengl_debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

}

#endif
