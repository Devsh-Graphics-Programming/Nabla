#include "CVulkanPipelineLayout.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineLayout::~CVulkanPipelineLayout()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyPipelineLayout(device, m_layout, nullptr);
    }
}

}