#include "CVulkanBuffer.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanBuffer::~CVulkanBuffer()
{
    auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = reinterpret_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyBuffer(device, m_vkBuffer, nullptr);
    }
}

}