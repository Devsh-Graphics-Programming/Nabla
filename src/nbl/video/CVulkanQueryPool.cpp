#include "nbl/video/CVulkanQueryPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanQueryPool::~CVulkanQueryPool()
{
    if(VK_NULL_HANDLE != m_queryPool)
    {
        const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
        auto* vk = vulkanDevice->getFunctionTable();
        vk->vk.vkDestroyQueryPool(vulkanDevice->getInternalObject(), m_queryPool, nullptr);
    }
}

}