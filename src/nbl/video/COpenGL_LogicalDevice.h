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
        IOpenGL_LogicalDevice(_egl, config, major, minor),
        m_threadHandler(_egl, _features, config, major, minor),
        m_thread(&CThreadHandler<FunctionTableType>::thread, &m_threadHandler)
    {
        EGLContext master_ctx = m_threadHandler.getContext();
        uint32_t totalQCount = 0u;
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
            totalQCount += params.queueCreateInfos[i].count;
        assert(totalQCount <= MaxQueueCount);

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
                (*m_queues)[ix] = core::make_smart_refctd_ptr<QueueType>(_egl, _features, master_ctx, config, famIx, flags, priority);
            }
        }
    }

    ~COpenGL_LogicalDevice()
    {
        m_threadHandler.terminate(m_thread);
    }

private:
    CThreadHandler<FunctionTableType> m_threadHandler;
    std::thread m_thread;
};

}
}

#endif