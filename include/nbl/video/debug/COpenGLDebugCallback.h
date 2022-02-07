#ifndef __NBL_VIDEO_C_OPENGL_DEBUG_CALLBACK_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_DEBUG_CALLBACK_H_INCLUDED__

#include "nbl/video/debug/IDebugCallback.h"

namespace nbl::video
{
class COpenGLDebugCallback : public IDebugCallback
{
public:
    COpenGLDebugCallback()
        : IDebugCallback(nullptr), m_callback(nullptr) {}
    explicit COpenGLDebugCallback(core::smart_refctd_ptr<system::ILogger>&& _logger)
        : IDebugCallback(std::move(_logger)), m_callback(&defaultCallback) {}

    // avoiding OpenGL typedefs here
    static void defaultCallback(uint32_t source, uint32_t type, uint32_t id, uint32_t severity, int32_t length, const char* message, const void* userParam);

    decltype(&defaultCallback) m_callback;
};

}

#endif
