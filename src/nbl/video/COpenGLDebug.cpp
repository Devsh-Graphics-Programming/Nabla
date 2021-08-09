#include "nbl/video/COpenGLDebug.h"

namespace nbl::video
{

void opengl_debug_callback(GLenum _source, GLenum _type, GLuint _id, GLenum _severity, GLsizei _length, const GLchar* _message, const void* _userParam)
{
    const SDebugCallback* cb = reinterpret_cast<const SDebugCallback*>(_userParam);
    E_DEBUG_MESSAGE_SEVERITY severity = EDMS_VERBOSE;
    switch (_severity)
    {
    case GL_DEBUG_SEVERITY_HIGH:
        severity = EDMS_ERROR; break;
    case GL_DEBUG_SEVERITY_MEDIUM: [[fallthrough]];
    case GL_DEBUG_SEVERITY_LOW:
        severity = EDMS_WARNING; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        severity = EDMS_INFO; break;
    }
    E_DEBUG_MESSAGE_TYPE type = EDMT_GENERAL;
    switch (_type)
    {
    case GL_DEBUG_TYPE_ERROR: [[fallthrough]];
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: [[fallthrough]];
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: [[fallthrough]];
    case GL_DEBUG_TYPE_PORTABILITY:
        type = EDMT_VALIDATION; break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        type = EDMT_PERFORMANCE; break;
    case GL_DEBUG_TYPE_OTHER:
    case GL_DEBUG_TYPE_MARKER:
        type = EDMT_GENERAL; break;
    }

    cb->callback(severity, type, _message, cb->userData);
}

}