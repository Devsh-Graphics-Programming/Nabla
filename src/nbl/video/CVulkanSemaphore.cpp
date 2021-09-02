#include "nbl/video/CVulkanSemaphore.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanSemaphore::~CVulkanSemaphore()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroySemaphore(vkdev, m_semaphore, nullptr);
        vkDestroySemaphore(device, m_semaphore, nullptr);
    }
}

}