#include "CVulkanMemoryAllocation.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanMemoryAllocation::~CVulkanMemoryAllocation()
{
    VkDevice device = static_cast<const CVulkanLogicalDevice*>(m_originDevice)->getInternalObject();
    vkFreeMemory(device, m_deviceMemoryHandle, nullptr);
}

}