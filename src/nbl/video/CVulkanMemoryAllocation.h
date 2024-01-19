#ifndef _NBL_VIDEO_C_VULKAN_MEMORY_ALLOCATION_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_MEMORY_ALLOCATION_H_INCLUDED_


#include "nbl/video/IDeviceMemoryAllocation.h"

#include <volk.h>


namespace nbl::video
{
class CVulkanLogicalDevice;

class CVulkanMemoryAllocation : public IDeviceMemoryAllocation
{
    public:
        CVulkanMemoryAllocation(
            const CVulkanLogicalDevice* dev,
            const VkDeviceMemory deviceMemoryHandle, 
            SCreationParams&& params
        );

        inline VkDeviceMemory getInternalObject() const { return m_deviceMemoryHandle; }

    private:
        ~CVulkanMemoryAllocation();

        void* map_impl(const MemoryRange& range, const core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> accessHint) override;
        bool unmap_impl() override;

        core::smart_refctd_ptr<const CVulkanLogicalDevice> m_vulkanDevice;
        const VkDeviceMemory m_deviceMemoryHandle;
};

}
#endif
