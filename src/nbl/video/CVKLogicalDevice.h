#ifndef __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"
#include "nbl/video/CVKSwapchain.h"
#include "nbl/video/CVulkanQueue.h"

namespace nbl {
namespace video
{

class CVKLogicalDevice final : public ILogicalDevice
{
public:
    CVKLogicalDevice(VkDevice vkdev) :
        m_vkdev(vkdev),
        m_devf(vkdev)
    {
        
    }

    ~CVKLogicalDevice()
    {
        m_devf.vk.vkDestroyDevice(m_vkdev, nullptr);
    }

    core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) override
    {
        return core::make_smart_refctd_ptr<CVKSwapchain>(std::move(params), this);
    }

    CVulkanDeviceFunctionTable* getFunctionTable() { return &m_devf; }
    VkDevice getInternalObject() const { return m_vkdev; }

private:
    VkDevice m_vkdev;
    CVulkanDeviceFunctionTable m_devf;
    using queues_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<CVulkanQueue>>;
    queues_array_t m_queues;
};

}
}

#endif