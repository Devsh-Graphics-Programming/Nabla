#include "CVulkanBufferView.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanBufferView::~CVulkanBufferView()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroyBufferView(vkdev, m_vkBufferView, nullptr);
        vkDestroyBufferView(device, m_vkBufferView, nullptr);
    }
}

}