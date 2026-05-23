#include "nbl/video/CVulkanMemoryAllocation.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
CVulkanMemoryAllocation::CVulkanMemoryAllocation(
    const CVulkanLogicalDevice* dev, 
    const VkDeviceMemory deviceMemoryHandle,
    std::unique_ptr<system::external_handle_t[]> externalHandles,
    SCreationParams&& params
) : IDeviceMemoryAllocation(dev,std::move(params)), m_vulkanDevice(dev), m_deviceMemoryHandle(deviceMemoryHandle), m_externalHandles(std::move(externalHandles)) {}

system::external_handle_t CVulkanMemoryAllocation::getExportHandle(E_EXTERNAL_HANDLE_TYPE handleType) const
{
  using U = typename core::bitflag<E_EXTERNAL_HANDLE_TYPE>::UNDERLYING_TYPE;

  if (!std::has_single_bit(static_cast<U>(handleType))) return nullptr;

  const auto externalHandleTypes = getCreationParams().externalHandleTypes;
  if (!externalHandleTypes.hasFlags(handleType)) return nullptr;

  const auto mask = core::bitflag<E_EXTERNAL_HANDLE_TYPE>(handleType - 1);
  const auto handleIndex = hlsl::bitCount(externalHandleTypes & mask);

  return m_externalHandles[handleIndex];
}

CVulkanMemoryAllocation::~CVulkanMemoryAllocation()
{
    if (m_externalHandles != nullptr)
    {
        const auto externalHandleCount = hlsl::bitCount(getCreationParams().externalHandleTypes);

        for (auto i = 0; i < externalHandleCount; i++)
        {
            const auto externalHandle = m_externalHandles[i];
            const auto success = system::CloseExternalHandle(externalHandle);
            if (!success) m_vulkanDevice->getLogger()->log("Failed to close external handle for Vulkan memory allocation", system::ILogger::ELL_ERROR);
        }
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