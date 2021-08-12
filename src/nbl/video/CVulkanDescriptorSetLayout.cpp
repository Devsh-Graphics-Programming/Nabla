#include "CVulkanDescriptorSetLayout.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorSetLayout::~CVulkanDescriptorSetLayout()
{
    auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyDescriptorSetLayout(device, m_dsLayout, nullptr);
    }
}

}