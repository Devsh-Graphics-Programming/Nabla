#ifndef __NBL_C_VULKAN_QUEUE_H_INCLUDED__
#define __NBL_C_VULKAN_QUEUE_H_INCLUDED__

#include <volk.h>
#include "nbl/video/IGPUQueue.h"

namespace nbl::video
{

class CVKLogicalDevice;

class CVulkanQueue final : public IGPUQueue
{
public:
    CVulkanQueue(CVKLogicalDevice* vkdev, VkQueue vkq, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority);

    bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override;

    // This API needs to change, we need more granularity than just saying if presentation
    // failed or succeeded
    bool present(const SPresentInfo& info) override;

    inline VkQueue getInternalObject() const { return m_vkqueue; }

private:
    CVKLogicalDevice* m_vkdevice;
    VkQueue m_vkqueue;
};

}

#endif
