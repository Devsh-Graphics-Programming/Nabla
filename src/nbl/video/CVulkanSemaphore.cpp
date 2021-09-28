#include "nbl/video/CVulkanSemaphore.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanSemaphore::~CVulkanSemaphore()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroySemaphore(vulkanDevice->getInternalObject(), m_semaphore, nullptr);
}

}