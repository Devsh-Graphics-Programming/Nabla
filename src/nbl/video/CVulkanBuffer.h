#ifndef __NBL_C_VULKAN_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUBuffer.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanBuffer : public IGPUBuffer
{
public:
    CVulkanBuffer(ILogicalDevice* dev, const IDriverMemoryBacked::SDriverMemoryRequirements& reqs, 
        VkBuffer buffer) : IGPUBuffer(dev, reqs), m_buffer(buffer)
    {}

    ~CVulkanBuffer();

    inline VkBuffer getInternalObject() const { return m_buffer; };

    // Todo(achal): I don't think its possible
    bool canUpdateSubRange() const override { return false; }

    //! Returns the allocation which is bound to the resource
    IDriverMemoryAllocation* getBoundMemory() override
    {
        return nullptr;
    }

    //! Constant version
    const IDriverMemoryAllocation* getBoundMemory() const override
    {
        return nullptr;
    }

    //! Returns the offset in the allocation at which it is bound to the resource
    size_t getBoundMemoryOffset() const override
    {
        return 0ull;
    }

private:
    // Todo(achal): A smart_refctd_ptr to buffer's memory, perhaps?
    VkBuffer m_buffer;

};

}

#define __NBL_C_VULKAN_BUFFER_H_INCLUDED__
#endif
