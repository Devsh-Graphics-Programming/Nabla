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
    constexpr static inline size_t MIN_ALLOC_SZ = 64ull;
    // TODO: there's an optimization possible if we set the block size to 64kb (max cmd upload size) and then:
    // - use Pool Address Allocator within a block
    // - allocate new block whenever we want to update buffer via cmdbuf
    // - set the pool's brick size to the maximum data store by any command possible
    // - when a command needs an unbounded variable length list of arguments, store it via linked list (chunk it)
    //constexpr static inline size_t MAX_COMMAND_STORAGE_SZ = 32ull;
    constexpr static inline size_t BLOCK_SIZE = 1ull << 21u;
    constexpr static inline size_t MAX_BLOCK_COUNT = 256u;

public:
    using IGPUCommandPool::IGPUCommandPool;
    COpenGLCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<E_CREATE_FLAGS> _flags, uint32_t _familyIx)
        : IGPUCommandPool(std::move(dev), _flags.value, _familyIx), mempool(BLOCK_SIZE, 0u, MAX_BLOCK_COUNT, MIN_ALLOC_SZ) {}

    template<typename T, typename... Args>
    T* emplace_n(uint32_t n, Args&&... args)
    {
        //static_assert(n*sizeof(T)<=MAX_COMMAND_STORAGE_SZ,"Command Data Store Type larger than preset limit!");
        std::unique_lock<std::mutex> lk(mutex);
        return mempool.emplace_n<T>(n, std::forward<Args>(args)...);
    }
    template<typename T>
    void free_n(const T* ptr, uint32_t n)
    {
        //static_assert(n*sizeof(T)<=MAX_COMMAND_STORAGE_SZ,"Command Data Store Type larger than preset limit!");
        std::unique_lock<std::mutex> lk(mutex);
        mempool.free_n<T>(const_cast<T*>(ptr), n);
    }

private:
    std::mutex mutex;
    core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>, core::default_aligned_allocator, false, uint32_t> mempool;
};

}

#endif
