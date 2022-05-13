#ifndef __NBL_C_VULKAN_MEMORY_ALLOCATION_H_INCLUDED__

#include "nbl/video/IDriverMemoryAllocation.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanMemoryAllocation : public IDriverMemoryAllocation
{
public:
    CVulkanMemoryAllocation(ILogicalDevice* dev, size_t size, bool isDedicated, VkDeviceMemory deviceMemoryHandle, core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> flags = E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE)
        : IDriverMemoryAllocation(dev, flags), m_size(size), m_isDedicated(isDedicated), m_deviceMemoryHandle(deviceMemoryHandle)
    {}

    ~CVulkanMemoryAllocation();

    //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
    bool isDedicated() const override { return m_isDedicated; }

    //! Returns the size of the memory allocation
    size_t getAllocationSize() const override { return m_size; }

    inline VkDeviceMemory getInternalObject() const { return m_deviceMemoryHandle; }

    inline void setMembersPostMap(void* ptr, MemoryRange rng)
    {
        // Probably should just make this public
        postMapSetMembers(ptr, rng);
    }

private:
    VkDeviceMemory m_deviceMemoryHandle;
    const bool m_isDedicated;
    const size_t m_size;
};

}

#define __NBL_C_VULKAN_MEMORY_ALLOCATION_H_INCLUDED__
#endif
