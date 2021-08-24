#include "CVulkanMemoryAllocation.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanMemoryAllocation::~CVulkanMemoryAllocation()
{
    if (m_originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVulkanLogicalDevice*>(m_originDevice)->getInternalObject();
        vkFreeMemory(device, m_deviceMemoryHandle, nullptr);
    }
}

}