#ifndef __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CEGLCaller.h"
#include "nbl/video/COpenGLFunctionTable.h"
#include "nbl/video/COpenGLQueue.h"

namespace nbl {
namespace video
{

class COpenGLLogicalDevice final : public ILogicalDevice
{
public:
    COpenGLLogicalDevice(egl::CEGLCaller* _egl, const SCreationParams& params) :
        ILogicalDevice(params),
        m_gl(_egl)
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
                (*m_queues)[ix] = core::make_smart_refctd_ptr<COpenGLQueue>(&m_gl, famIx, flags, priority);
            }
        }
    }

private:
    COpenGLFunctionTable m_gl;
};

}
}

#endif