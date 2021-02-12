#ifndef __NBL_C_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl {
namespace video
{

template <typename QueueType_>
class COpenGL_LogicalDevice final : public IOpenGL_LogicalDevice
{
public:
    using QueueType = QueueType_;
    using FunctionTableType = typename QueueType::FunctionTableType;
    using FeaturesType = typename QueueType::FeaturesType;

    COpenGL_LogicalDevice(const egl::CEGL* _egl, FeaturesType* _features, EGLConfig config, EGLint major, EGLint minor, const SCreationParams& params) :
        IOpenGL_LogicalDevice(_egl, config, major, minor)
    {
        EGLint ctx_attributes[] = {
            EGL_CONTEXT_MAJOR_VERSION, major,
            EGL_CONTEXT_MINOR_VERSION, minor,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

            EGL_NONE
        };
        // master context for queues
        // TODO: take the one from device thread handler
        m_ctx = _egl->call.peglCreateContext(_egl->display, config, EGL_NO_CONTEXT, ctx_attributes);

        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueCreateInfos[i];
            const uint32_t famIx = qci.familyIndex;
            const uint32_t offset = (*m_offsets)[famIx];
            const auto flags = qci.flags;

            for (uint32_t j = 0u; j < qci.count; ++j)
            {
                const float priority = qci.priorities[j];

                const uint32_t ix = offset + j;
                (*m_queues)[ix] = core::make_smart_refctd_ptr<QueueType>(_egl, _features, m_ctx, config, famIx, flags, priority);
            }
        }
    }

private:
    // context used to run GL calls from logical device (mostly resource creation) in separate thread;
    // also being master context for all the other ones (the ones in swapchains and queues)
    EGLContext m_ctx;
    // TODO pbuf surface
};

}
}

#endif