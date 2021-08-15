#ifndef __NBL_C_VULKAN_QUEUE_H_INCLUDED__
#define __NBL_C_VULKAN_QUEUE_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IGPUQueue.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanQueue final : public IGPUQueue
{
public:
    CVulkanQueue(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, VkQueue vkq, uint32_t _famIx,
        E_CREATE_FLAGS _flags, float _priority) : IGPUQueue(std::move(logicalDevice), _famIx,
            _flags, _priority), m_vkQueue(vkq)
    {}

    bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override;

    // This API needs to change, we need more granularity than just saying if presentation
    // failed or succeeded
    bool present(const SPresentInfo& info) override;

    inline VkQueue getInternalObject() const { return m_vkQueue; }

private:
    VkQueue m_vkQueue;
};

}

#endif
