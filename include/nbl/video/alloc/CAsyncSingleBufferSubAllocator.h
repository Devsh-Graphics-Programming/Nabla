// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_ASYNC_SINGLE_BUFFER_SUB_ALLOCATOR_H_
#define _NBL_VIDEO_C_ASYNC_SINGLE_BUFFER_SUB_ALLOCATOR_H_


#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"
#include "nbl/video/alloc/CSingleBufferSubAllocator.h"
#include "nbl/video/TimelineEventHandlers.h"

#include <mutex>


namespace nbl::video
{

namespace impl
{
// HostAllocator allocates both reserved space and the space needed for variable length records on the DeferredFreeFunctor
template<class AddressAllocator, class HostAllocator>
class CAsyncSingleBufferSubAllocator
{
    private:
        #ifdef _NBL_DEBUG
        std::recursive_mutex stAccessVerfier;
        #endif // _NBL_DEBUG
        using ThisType = CAsyncSingleBufferSubAllocator<AddressAllocator,HostAllocator>;
        using Composed = CSingleBufferSubAllocator<AddressAllocator,HostAllocator>;
        
    public:
        using size_type = typename AddressAllocator::size_type;
        using value_type = typename Composed::value_type;
        static constexpr value_type invalid_value = Composed::invalid_value;

        class DeferredFreeFunctor
        {
                using ref_t = core::smart_refctd_ptr<const core::IReferenceCounted>;
                static constexpr size_t PseudoTupleByteSize = 2u*sizeof(size_type)+sizeof(ref_t);
                static constexpr size_t AllocatorUnitsPerMetadata = PseudoTupleByteSize/sizeof(typename HostAllocator::value_type);
                static_assert((PseudoTupleByteSize%sizeof(typename HostAllocator::value_type)) == 0u, "should be divisible by HostAllocator::value_type");

            public:
                template<typename T> requires std::is_base_of_v<core::IReferenceCounted,T>
                inline DeferredFreeFunctor(Composed* _composed, size_type numAllocsToFree, const size_type* addrs, const size_type* bytes, const T*const *const objectsToHold)
                    : composed(_composed), rangeData(nullptr), numAllocs(numAllocsToFree)
                {
                    static_assert(std::is_base_of_v<core::IReferenceCounted,T>);

                    // TODO : CMemoryPool<RobustGeneralpurposeAllocator> a-la naughty dog
                    rangeData = reinterpret_cast<size_type*>(composed->getReservedAllocator().allocate(AllocatorUnitsPerMetadata*numAllocs,sizeof(void*)));
                    auto out = rangeData;
                    memcpy(out,addrs,sizeof(size_type)*numAllocs);
                    out += numAllocs;
                    memcpy(out,bytes,sizeof(size_type)*numAllocs);
                    out += numAllocs;
                    for (size_t i=0u; i<numAllocs; i++)
                        std::construct_at<ref_t>(reinterpret_cast<ref_t*>(out)+i,objectsToHold ? objectsToHold[i]:nullptr);
                }
                DeferredFreeFunctor(const DeferredFreeFunctor& other) = delete;
                inline DeferredFreeFunctor(DeferredFreeFunctor&& other) : composed(nullptr), rangeData(nullptr), numAllocs(0u)
                {
                    operator=(std::move(other));
                }

                inline ~DeferredFreeFunctor()
                {
                    if (rangeData)
                        composed->getReservedAllocator().deallocate(reinterpret_cast<typename HostAllocator::pointer>(rangeData),AllocatorUnitsPerMetadata*numAllocs);
                }

                DeferredFreeFunctor& operator=(const DeferredFreeFunctor& other) = delete;
                inline DeferredFreeFunctor& operator=(DeferredFreeFunctor&& other)
                {
                    if (rangeData) // could swap the values instead
                        composed->getReservedAllocator().deallocate(reinterpret_cast<typename HostAllocator::pointer>(rangeData),AllocatorUnitsPerMetadata*numAllocs);
                    composed = other.composed;
                    rangeData = other.rangeData;
                    numAllocs = other.numAllocs;
                    other.composed = nullptr;
                    other.rangeData = nullptr;
                    other.numAllocs = 0u;
                    return *this;
                }

                inline bool operator()(size_type& unallocatedSize)
                {
                    operator()();
                    for (size_type i=0u; i<numAllocs; i++)
                    {
                        auto freedSize = rangeData[numAllocs+i];
                        if (unallocatedSize>freedSize)
                            unallocatedSize -= freedSize;
                        else
                        {
                            unallocatedSize = 0u;
                            return true;
                        }
                    }
                    return unallocatedSize==0u;
                }

                inline void operator()()
                {
                    std::destroy_n(reinterpret_cast<ref_t*>(rangeData+numAllocs*2u),numAllocs);
                    #ifdef _NBL_DEBUG
                    assert(composed && rangeData);
                    #endif // _NBL_DEBUG
                    composed->multi_deallocate(numAllocs,rangeData,rangeData+numAllocs);
                }

            private:
                Composed* composed;
                size_type* rangeData;
                size_type numAllocs;
        };

        // perfect forward ctor to `CSingleBufferSubAllocator`
        template<typename... Args>
        inline CAsyncSingleBufferSubAllocator(Args&&... args) : m_composed(std::forward<Args>(args)...),
            deferredFrees(const_cast<ILogicalDevice*>(m_composed.getBuffer()->getOriginDevice())) {}
        virtual ~CAsyncSingleBufferSubAllocator() {}


        // anyone gonna use it?
        inline const AddressAllocator& getAddressAllocator() const {return m_composed.getAddressAllocator();}

        // buffer getters
        inline IGPUBuffer* getBuffer() {return m_composed.getBuffer();}
        inline const IGPUBuffer* getBuffer() const {return m_composed.getBuffer();}

        //! Returns free events still outstanding
        inline uint32_t cull_frees() noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            return deferredFrees.poll().eventsLeft;
        }

        //! Returns max possible currently allocatable single allocation size, without having to wait for GPU more
        inline size_type max_size() noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            size_type valueToStopAt = getAddressAllocator().min_size()*3u; // padding, allocation, more padding = 3u
            // we don't actually want or need to poll all possible blocks to free, only first few
            deferredFrees.poll(valueToStopAt);
            return getAddressAllocator().max_size();
        }


        //! allocate with default defragmentation timeout
        template<typename... Args>
        inline size_type multi_allocate(uint32_t count, Args&&... args) noexcept
        {
            return multi_alloc(decltype(deferredFrees)::default_wait(),count,std::forward<Args>(args)...);
        }
        //! attempt to allocate, if fail (presumably because of fragmentation), then keep trying till timeout is reached
        template<class Clock=typename std::chrono::steady_clock, typename... Args>
        inline size_type multi_allocate(const std::chrono::time_point<Clock>& maxWaitPoint, const Args&... args) noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG

            // try allocate once
            size_type unallocatedSize = try_multi_alloc(args...);
            if (!unallocatedSize)
                return 0u;

            // then try to wait at least once and allocate
            do
            {
                deferredFrees.wait(maxWaitPoint,unallocatedSize);

                unallocatedSize = try_multi_alloc(args...);
                if (!unallocatedSize)
                    return 0u;
            } while(Clock::now()<maxWaitPoint);

            return unallocatedSize;
        }

        //!
        inline void multi_deallocate(const ISemaphore::SWaitInfo& futureWait, DeferredFreeFunctor&& functor) noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            deferredFrees.latch(futureWait,std::move(functor));
        }
        inline void multi_deallocate(uint32_t count, const value_type* addr, const size_type* bytes) noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            multi_deallocate(count,addr,bytes,{});
        }
        // TODO: improve signature of this function in the future
        template<typename T=core::IReferenceCounted>
        inline void multi_deallocate(uint32_t count, const value_type* addr, const size_type* bytes, const ISemaphore::SWaitInfo& futureWait, const T*const *const objectsToDrop=nullptr) noexcept
        {
            if (futureWait.semaphore)
                multi_deallocate(futureWait,DeferredFreeFunctor(&m_composed,count,addr,bytes,objectsToDrop));
            else
                multi_deallocate(count,addr,bytes);
        }

    protected:
        Composed m_composed;
        MultiTimelineEventHandlerST<DeferredFreeFunctor> deferredFrees;

        template<typename... Args>
        inline value_type try_multi_alloc(uint32_t count, value_type* outAddresses, const size_type* byteSizes, const Args&... args) noexcept
        {
            m_composed.multi_allocate(count,outAddresses,byteSizes,args...);

            size_type unallocatedSize = 0;
            for (uint32_t i=0u; i<count; i++)
            {
                if (outAddresses[i]!=invalid_value)
                    continue;

                unallocatedSize += byteSizes[i];
            }
            return unallocatedSize;
        }
};
}

// this buffer is not growable
template<class AddressAllocator=core::GeneralpurposeAddressAllocator<uint32_t>, class HostAllocator=core::allocator<uint8_t>>
class CAsyncSingleBufferSubAllocatorST final : public core::IReferenceCounted, public impl::CAsyncSingleBufferSubAllocator<AddressAllocator,HostAllocator>
{
        using Base = impl::CAsyncSingleBufferSubAllocator<AddressAllocator,HostAllocator>;

    protected:
        ~CAsyncSingleBufferSubAllocatorST() = default;

    public:
        template<typename... Args>
        CAsyncSingleBufferSubAllocatorST(Args&&... args) : Base(std::forward<Args>(args)...) {}
};

//MT version?

}

#endif




