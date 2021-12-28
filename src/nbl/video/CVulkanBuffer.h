#ifndef __NBL_C_VULKAN_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUBuffer.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;
class CVulkanMemoryAllocation;

class CVulkanBuffer : public IGPUBuffer
{
public:
    CVulkanBuffer(
        core::smart_refctd_ptr<ILogicalDevice>&& dev,
        const IDriverMemoryBacked::SDriverMemoryRequirements& reqs,
        const IGPUBuffer::SCachedCreationParams& cachedCreationParams, VkBuffer buffer
    ) : IGPUBuffer(std::move(dev),reqs,cachedCreationParams), m_vkBuffer(buffer)
    {
    }

    ~CVulkanBuffer();

    inline VkBuffer getInternalObject() const { return m_vkBuffer; };

    IDriverMemoryAllocation* getBoundMemory() override
    {
        return m_memory.get();
    }

    const IDriverMemoryAllocation* getBoundMemory() const override
    {
        return m_memory.get();
    }

    size_t getBoundMemoryOffset() const override
    {
        return m_memBindingOffset;
    }

    inline void setMemoryAndOffset(core::smart_refctd_ptr<IDriverMemoryAllocation>&& memory, uint64_t memBindingOffset)
    {
        m_memory = std::move(memory);
        m_memBindingOffset = memBindingOffset;
    }
    
    void setObjectDebugName(const char* label) const override;

private:
    core::smart_refctd_ptr<IDriverMemoryAllocation> m_memory = nullptr;
    uint64_t m_memBindingOffset;
    VkBuffer m_vkBuffer;
    uint64_t m_bufferSize;
};

}

#define __NBL_C_VULKAN_BUFFER_H_INCLUDED__
#endif
