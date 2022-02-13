#include "nbl/video/CVulkanMemoryAllocation.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanMemoryAllocation::~CVulkanMemoryAllocation()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(m_originDevice);
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkFreeMemory(vulkanDevice->getInternalObject(), m_deviceMemoryHandle, nullptr);
}

}