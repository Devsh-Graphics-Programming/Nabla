#ifndef __NBL_C_VULKAN_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"

#include <mutex>

#include <volk.h>

namespace nbl::video
{

class CVulkanCommandPool final : public IGPUCommandPool
{
     constexpr static inline uint32_t NODES_PER_BLOCK = 4096u;
     constexpr static inline uint32_t MAX_BLOCK_COUNT = 256u;

public:
    struct ArgumentReferenceSegment
    {
        ArgumentReferenceSegment() : arguments(), argCount(0u), next(nullptr) {}
    
        constexpr static uint8_t MAX_REFERENCES = 62u;
        std::array<core::smart_refctd_ptr<const core::IReferenceCounted>, MAX_REFERENCES> arguments;

        uint8_t argCount;
        ArgumentReferenceSegment* next;
    };

    CVulkanCommandPool(core::smart_refctd_ptr<ILogicalDevice>&& dev,
        core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags, uint32_t queueFamilyIndex,
        VkCommandPool vk_commandPool)
        : IGPUCommandPool(std::move(dev), flags.value, queueFamilyIndex),
        m_vkCommandPool(vk_commandPool), mempool(NODES_PER_BLOCK * sizeof(ArgumentReferenceSegment),
            1u, MAX_BLOCK_COUNT, static_cast<uint32_t>(sizeof(ArgumentReferenceSegment)))
    {}

    void emplace_n(ArgumentReferenceSegment*& tail,
        const core::smart_refctd_ptr<const core::IReferenceCounted>* begin,
        const core::smart_refctd_ptr<const core::IReferenceCounted>* end)
    {
        if (!tail)
            tail = mempool.emplace<ArgumentReferenceSegment>();

        auto it = begin;
        while (it != end)
        {
            // allocate new segment if overflow
            if (tail->argCount == ArgumentReferenceSegment::MAX_REFERENCES)
            {
                auto newTail = mempool.emplace<ArgumentReferenceSegment>();
                tail->next = newTail;
                tail = newTail;
            }

            // fill to the brim
            const auto count = core::min(end - it, ArgumentReferenceSegment::MAX_REFERENCES - tail->argCount);
            std::copy_n(it, count, tail->arguments.begin() + tail->argCount);
            it += count;
            tail->argCount += count;
        }
    }

    void free_all(ArgumentReferenceSegment* head)
    {
        while (head)
        {
            ArgumentReferenceSegment* next = head->next;
            mempool.free<ArgumentReferenceSegment>(head);
            head = next;
        }
    }

    VkCommandPool getInternalObject() const { return m_vkCommandPool; }

    ~CVulkanCommandPool();

private:
    VkCommandPool m_vkCommandPool;
    core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, core::default_aligned_allocator, uint32_t> mempool;
};

}
#define __NBL_C_VULKAN_COMMAND_POOL_H_INCLUDED__
#endif
