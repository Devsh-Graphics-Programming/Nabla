#include "CVulkanDescriptorPool.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorPool::~CVulkanDescriptorPool()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    }
}

}