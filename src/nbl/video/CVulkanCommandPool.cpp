#include "CVulkanCommandPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanCommandPool::~CVulkanCommandPool()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyCommandPool(vulkanDevice->getInternalObject(), m_vkCommandPool, nullptr);
}

}