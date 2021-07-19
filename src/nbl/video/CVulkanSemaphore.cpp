#include "nbl/video/CVulkanSemaphore.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

CVulkanSemaphore::CVulkanSemaphore(CVKLogicalDevice* _vkdev) : IGPUSemaphore(_vkdev), m_vkdev(_vkdev)
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    VkSemaphoreCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = static_cast<VkSemaphoreCreateFlags>(0);
    vk->vk.vkCreateSemaphore(vkdev, &ci, nullptr, &m_semaphore);
}

CVulkanSemaphore::~CVulkanSemaphore()
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    vk->vk.vkDestroySemaphore(vkdev, m_semaphore, nullptr);
}

}
}