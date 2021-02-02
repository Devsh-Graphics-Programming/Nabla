#ifndef __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CEGL.h"
#include "nbl/video/COpenGLQueue.h"

namespace nbl {
namespace video
{

class COpenGLLogicalDevice final : public ILogicalDevice
{
public:
    COpenGLLogicalDevice(const egl::CEGL* _egl, EGLContext master_ctx, EGLConfig config, const SCreationParams& params) :
        ILogicalDevice(params)
    {
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
                // TODO will also have to pass master context to the queue and config
                (*m_queues)[ix] = core::make_smart_refctd_ptr<COpenGLQueue>(_egl, master_ctx, config, famIx, flags, priority);
            }
        }
    }
};

}
}

#endif