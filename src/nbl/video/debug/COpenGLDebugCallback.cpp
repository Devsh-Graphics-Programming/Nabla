#include "nbl/video/debug/COpenGLDebugCallback.h"
#include "nbl/video/COpenGLFunctionTable.h"

using namespace nbl;
using namespace nbl::video;

void COpenGLDebugCallback::defaultCallback(GLenum _source, GLenum _type, GLuint _id, GLenum _severity, GLsizei _length, const GLchar* _message, const void* _userParam)
{
    const auto* cb = reinterpret_cast<const COpenGLDebugCallback*>(_userParam);
    uint8_t level = 0u;
    switch (_severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:
            level |= system::ILogger::ELL_PERFORMANCE;
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            level |= system::ILogger::ELL_WARNING;
            break;
        case GL_DEBUG_SEVERITY_LOW:
            level |= system::ILogger::ELL_INFO;
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            break;
    }
    switch (_type)
    {
        case GL_DEBUG_TYPE_ERROR:
            [[fallthrough]];
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            level |= system::ILogger::ELL_ERROR;
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            [[fallthrough]];
        case GL_DEBUG_TYPE_PORTABILITY:
            level |= system::ILogger::ELL_WARNING;
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            level |= system::ILogger::ELL_PERFORMANCE;
            break;
        case GL_DEBUG_TYPE_OTHER:
            level |= system::ILogger::ELL_INFO;
            break;
        case GL_DEBUG_TYPE_MARKER:
            level |= system::ILogger::ELL_DEBUG;
            break;
    }
    level = 1 << core::findMSB(uint32_t(level));
    cb->getLogger()->log("%s",static_cast<system::ILogger::E_LOG_LEVEL>(level),_message);
}