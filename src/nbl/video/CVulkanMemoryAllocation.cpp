#include "CVulkanMemoryAllocation.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanMemoryAllocation::~CVulkanMemoryAllocation()
{
    if (m_originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVKLogicalDevice*>(m_originDevice)->getInternalObject();
        vkFreeMemory(device, m_deviceMemoryHandle, nullptr);
    }
}

}