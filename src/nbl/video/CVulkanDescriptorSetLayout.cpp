#include "CVulkanDescriptorSetLayout.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorSetLayout::~CVulkanDescriptorSetLayout()
{
    auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyDescriptorSetLayout(device, m_dsLayout, nullptr);
    }
}

}