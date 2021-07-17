#ifndef __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"
#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"
#include <mutex>

namespace nbl::video
{

class COpenGLCommandPool final : public IGPUCommandPool
{
        // TODO: tune these variables
        constexpr static inline size_t BLOCK_SIZE = 4096u*1024u;
        constexpr static inline size_t MAX_BLOCK_COUNT = 256u;

    public:
        using IGPUCommandPool::IGPUCommandPool;
        COpenGLCommandPool(ILogicalDevice* dev, E_CREATE_FLAGS _flags, uint32_t _familyIx) : IGPUCommandPool(dev, _flags, _familyIx), mempool(BLOCK_SIZE, MAX_BLOCK_COUNT) {}

        template <typename T, typename... Args>
        T* emplace_n(uint32_t n, Args&&... args)
        {
            std::unique_lock<std::mutex> lk(mutex);
            return mempool.emplace_n<T>(n, std::forward<Args>(args)...);
        }
        template <typename T>
        void free_n(const T* ptr, uint32_t n)
        {
            std::unique_lock<std::mutex> lk(mutex);
            mempool.free_n<T>(const_cast<T*>(ptr), n);
        }

    private:
        std::mutex mutex;
        core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>,core::default_aligned_allocator> mempool;
};

}

#endif
