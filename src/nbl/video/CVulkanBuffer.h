#ifndef __NBL_C_VULKAN_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUBuffer.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanBuffer : public IGPUBuffer
{
public:
    CVulkanBuffer(core::smart_refctd_ptr<ILogicalDevice>&& dev, const IDriverMemoryBacked::SDriverMemoryRequirements& reqs, 
        VkBuffer buffer) : IGPUBuffer(std::move(dev), reqs), m_vkBuffer(buffer)
    {}

    ~CVulkanBuffer();

    inline VkBuffer getInternalObject() const { return m_vkBuffer; };

    // Todo(achal): I don't think its possible
    bool canUpdateSubRange() const override { return false; }

    // Todo(achal)
    //! Returns the allocation which is bound to the resource
    IDriverMemoryAllocation* getBoundMemory() override
    {
        return nullptr;
    }

    // Todo(achal)
    //! Constant version
    const IDriverMemoryAllocation* getBoundMemory() const override
    {
        return nullptr;
    }

    // Todo(achal)
    //! Returns the offset in the allocation at which it is bound to the resource
    size_t getBoundMemoryOffset() const override
    {
        return 0ull;
    }

private:
    // Todo(achal): A smart_refctd_ptr to buffer's memory, perhaps?
    VkBuffer m_vkBuffer;
};

}

#define __NBL_C_VULKAN_BUFFER_H_INCLUDED__
#endif
