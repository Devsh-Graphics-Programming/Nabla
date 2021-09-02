#ifndef __NBL_C_VULKAN_MEMORY_ALLOCATION_H_INCLUDED__

#include "nbl/video/IDriverMemoryAllocation.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanMemoryAllocation : public IDriverMemoryAllocation
{
public:
    CVulkanMemoryAllocation(ILogicalDevice* dev, VkDeviceMemory deviceMemoryHandle)
        : IDriverMemoryAllocation(), m_originDevice(dev), m_deviceMemoryHandle(deviceMemoryHandle)
    {}

    ~CVulkanMemoryAllocation();

    //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
    bool isDedicated() const override { return false; }

    // Todo(achal)
    //! Returns the size of the memory allocation
    size_t getAllocationSize() const override { return 0ull; }

    inline VkDeviceMemory getInternalObject() const { return m_deviceMemoryHandle; }

private:
    VkDeviceMemory m_deviceMemoryHandle;
    ILogicalDevice* m_originDevice; // IDriverMemoryAllocation doesn't inherit from IBackendObject ???
};

}

#define __NBL_C_VULKAN_MEMORY_ALLOCATION_H_INCLUDED__
#endif
