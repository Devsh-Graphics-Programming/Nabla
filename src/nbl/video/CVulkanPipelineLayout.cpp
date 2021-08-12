#include "CVulkanPipelineLayout.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineLayout::~CVulkanPipelineLayout()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyPipelineLayout(device, m_layout, nullptr);
    }
}

}