#ifndef __NBL_C__VULKAN_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"

#include <mutex>

#include <volk.h>

namespace nbl::video
{

struct ArgumentReferenceSegment
{
    std::array<core::smart_refctd_ptr<core::IReferenceCounted>, 63> arguments;

    // What is this nextBlock here for?
    // What emplace would return?
    uint32_t argCount, nextBlock;
};

class CVulkanCommandPool final : public IGPUCommandPool
{
    constexpr static inline size_t BLOCK_SIZE = 4096u * 1024u;
    constexpr static inline size_t MAX_BLOCK_COUNT = 256u;


public:
    CVulkanCommandPool(ILogicalDevice* dev, IGPUCommandPool::E_CREATE_FLAGS flags,
        uint32_t queueFamilyIndex, VkCommandPool commandPool)
        : IGPUCommandPool(dev, flags, queueFamilyIndex), m_commandPool(commandPool),
        mempool(BLOCK_SIZE, MAX_BLOCK_COUNT)
    {}

#if 0
    template <typename T>
    void emplace_n(const uint32_t n, core::smart_refctd_ptr<T>* elements,
        ArgumentReferenceSegment* head, ArgumentReferenceSegment* tail)
    {
        std::unique_lock<std::mutex> lock(mutex);

        assert((head && tail) || (!head && !tail));

        if (!head)
        {
            mempool.emplace<ArgumentReferenceSegment>()
        }
        else
        {
            // We're currently at tail
            // How would I know if *tail is filled, so I need to ask a new segment from the pool?
            const uint32_t spaceLeftInCurrentSegment = 63 - tail->argCount;
            if (n > spaceLeftInCurrentSegment)
            {
                // ask a new segment from the pool
            }
            else
            {

            }

            finalTotal = tail->argCount + n
            if (tail->argCount < 63)

        }
        mempool.emplace
        // return mempool.emplace_n<T>()
    }
#endif

    VkCommandPool getInternalObject() const { return m_commandPool; }

    ~CVulkanCommandPool();

private:
    VkCommandPool m_commandPool;
    std::mutex mutex;
    core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, core::default_aligned_allocator> mempool;
};

}
#define __NBL_C__VULKAN_COMMAND_POOL_H_INCLUDED__
#endif
