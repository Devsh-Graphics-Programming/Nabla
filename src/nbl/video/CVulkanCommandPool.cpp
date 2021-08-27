#include "CVulkanCommandPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanCommandPool::~CVulkanCommandPool()
{
    auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = reinterpret_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyCommandPool(device, m_commandPool, nullptr);
    }
}

}