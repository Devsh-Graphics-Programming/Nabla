#include "nbl/video/CVulkanSemaphore.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanSemaphore::~CVulkanSemaphore()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroySemaphore(vkdev, m_semaphore, nullptr);
        vkDestroySemaphore(device, m_semaphore, nullptr);
    }
}

}