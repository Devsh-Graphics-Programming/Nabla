#include "nbl/video/CVulkanSemaphore.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanSemaphore::CVulkanSemaphore(CVKLogicalDevice* _vkdev, VkSemaphore semaphore)
    : IGPUSemaphore(_vkdev), m_vkdev(_vkdev), m_semaphore(semaphore)
{
}

CVulkanSemaphore::~CVulkanSemaphore()
{
    // auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    // vk->vk.vkDestroySemaphore(vkdev, m_semaphore, nullptr);
    vkDestroySemaphore(vkdev, m_semaphore, nullptr);
}

}