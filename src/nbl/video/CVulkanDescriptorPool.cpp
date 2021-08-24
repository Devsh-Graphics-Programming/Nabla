#include "CVulkanDescriptorPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorPool::~CVulkanDescriptorPool()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    }
}

}