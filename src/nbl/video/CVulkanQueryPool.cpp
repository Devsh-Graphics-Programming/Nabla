#include "nbl/video/CVulkanQueryPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanQueryPool::~CVulkanQueryPool()
{
    if(VK_NULL_HANDLE != m_queryPool)
    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
        auto* vk = vulkanDevice->getFunctionTable();
        VkDevice vk_device = vulkanDevice->getInternalObject();
        vk->vk.vkDestroyQueryPool(vk_device, m_queryPool, nullptr);
    }
}

}