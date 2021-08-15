#include "nbl/video/CVulkanFence.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanFence::~CVulkanFence()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroyFence(vkdev, m_fence, nullptr);
        vkDestroyFence(vk_device, m_fence, nullptr);
    }
}

}