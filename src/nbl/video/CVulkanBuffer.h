#ifndef _NBL_C_VULKAN_BUFFER_H_INCLUDED_
#define _NBL_C_VULKAN_BUFFER_H_INCLUDED_

#include "nbl/video/IGPUBuffer.h"

#define VK_NO_PROTOTYPES
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
            const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs,
            IGPUBuffer::SCreationParams&& creationParams, VkBuffer buffer
        ) : IGPUBuffer(std::move(dev),reqs,std::move(creationParams)), m_vkBuffer(buffer)
        {
            assert(m_vkBuffer != VK_NULL_HANDLE);
        }

        ~CVulkanBuffer();

        inline const void* getNativeHandle() const override {return &m_vkBuffer;}
        inline VkBuffer getInternalObject() const {return m_vkBuffer;}

        IDeviceMemoryAllocation* getBoundMemory() override
        {
            return m_memory.get();
        }

        const IDeviceMemoryAllocation* getBoundMemory() const override
        {
            return m_memory.get();
        }

        size_t getBoundMemoryOffset() const override
        {
            return m_memBindingOffset;
        }

        inline void setMemoryAndOffset(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& memory, uint64_t memBindingOffset)
        {
            m_memory = std::move(memory);
            m_memBindingOffset = memBindingOffset;
        }
    
        void setObjectDebugName(const char* label) const override;

    private:
        core::smart_refctd_ptr<IDeviceMemoryAllocation> m_memory = nullptr;
        uint64_t m_memBindingOffset;
        VkBuffer m_vkBuffer;
};

}

#endif
