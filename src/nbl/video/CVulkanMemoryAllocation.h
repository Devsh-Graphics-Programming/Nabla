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
            const external_handle_t externalHandle,
            SCreationParams&& params
        );

        inline VkDeviceMemory getInternalObject() const { return m_deviceMemoryHandle; }

        inline external_handle_t getExportHandle() const override
        {
          // Do not return duplicated importHandle
          if (m_params.importHandle == nullptr)
            return m_externalHandle;
          return nullptr;
        }

    private:
        ~CVulkanMemoryAllocation();

        void* map_impl(const MemoryRange& range, const core::bitflag<E_MAPPING_CPU_ACCESS_FLAGS> accessHint) override;
        bool unmap_impl() override;

        core::smart_refctd_ptr<const CVulkanLogicalDevice> m_vulkanDevice;
        const VkDeviceMemory m_deviceMemoryHandle;

        // Can store either duplicated importHandle or exportHandle.
        // This handle will be closed when destructor is called, unlike importHandle in SCreationParams.
        const external_handle_t m_externalHandle;
};

}
#endif
