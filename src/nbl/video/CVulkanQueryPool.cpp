#include "nbl/video/CVulkanQueryPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanQueryPool::~CVulkanQueryPool()
{
    if(VK_NULL_HANDLE != m_queryPool)
    {
        const auto originDevice = getOriginDevice();
        VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyQueryPool(vk_device, m_queryPool, nullptr);
    }
}

}