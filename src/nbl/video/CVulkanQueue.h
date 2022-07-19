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
    CVulkanQueue(ILogicalDevice* logicalDevice, renderdoc_api_t* rdoc, VkInstance vkinst, VkQueue vkq, uint32_t _famIx,
        E_CREATE_FLAGS _flags, float _priority)
        : IGPUQueue(logicalDevice, _famIx, _flags, _priority), m_vkQueue(vkq), m_rdoc_api(rdoc), m_vkInstance(vkinst)
    {}

    bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override;

    ISwapchain::E_PRESENT_RESULT present(const SPresentInfo& info) override;

    inline const void* getNativeHandle() const override {return &m_vkQueue;}
    inline VkQueue getInternalObject() const {return m_vkQueue;}

    bool startCapture() override;
    bool endCapture() override;

private:
    renderdoc_api_t* m_rdoc_api;
	VkInstance m_vkInstance;
    VkQueue m_vkQueue;
};

}

#endif
