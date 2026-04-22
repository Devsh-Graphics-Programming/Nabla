#include "nbl/video/CVulkanMemoryAllocation.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
CVulkanMemoryAllocation::CVulkanMemoryAllocation(
    const CVulkanLogicalDevice* dev, 
    const VkDeviceMemory deviceMemoryHandle,
    const external_handle_t externalHandle,
    SCreationParams&& params
) : IDeviceMemoryAllocation(dev,std::move(params)), m_vulkanDevice(dev), m_deviceMemoryHandle(deviceMemoryHandle), m_externalHandle(externalHandle) {}

CVulkanMemoryAllocation::~CVulkanMemoryAllocation()
{
    if (m_externalHandle != ExternalHandleNull)
    {
        bool re = CloseExternalHandle(m_externalHandle);
        assert(re);
    }
    m_vulkanDevice->getFunctionTable()->vk.vkFreeMemory(m_vulkanDevice->getInternalObject(),m_deviceMemoryHandle,nullptr);
}

void* CVulkanMemoryAllocation::map_impl(const MemoryRange& range, const core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> accessHint)
{
    void* retval = nullptr;
    const VkMemoryMapFlags vk_memoryMapFlags = 0; // reserved for future use, by Vulkan
    if (m_vulkanDevice->getFunctionTable()->vk.vkMapMemory(m_vulkanDevice->getInternalObject(),m_deviceMemoryHandle,range.offset,range.length,vk_memoryMapFlags,&retval)!=VK_SUCCESS)
        return nullptr;
    return retval;
}

bool CVulkanMemoryAllocation::unmap_impl()
{
    m_vulkanDevice->getFunctionTable()->vk.vkUnmapMemory(m_vulkanDevice->getInternalObject(),m_deviceMemoryHandle);
    return true;
}

}