// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_SUB_ALLOCATED_DATA_BUFFER_H__
#define __NBL_VIDEO_SUB_ALLOCATED_DATA_BUFFER_H__

#include "nbl/core/core.h"

#include <type_traits>
#include <mutex>

#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"
#include "nbl/video/IGPUFence.h"

namespace nbl::video
{

// this buffer is not growabl
template<class HeterogenousMemoryAddressAllocator, class CustomDeferredFreeFunctor=void>
class SubAllocatedDataBuffer : public virtual core::IReferenceCounted, protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor
{
    public:
        typedef typename HeterogenousMemoryAddressAllocator::OtherAllocatorType  GPUBufferAllocator;
        typedef typename HeterogenousMemoryAddressAllocator::HostAllocatorType  CPUAllocator;
        typedef typename HeterogenousMemoryAddressAllocator::size_type  size_type;
        static constexpr size_type invalid_address                                          = HeterogenousMemoryAddressAllocator::invalid_address;

    private:
        #ifdef _NBL_DEBUG
        std::recursive_mutex stAccessVerfier;
        #endif // _NBL_DEBUG
        typedef SubAllocatedDataBuffer<HeterogenousMemoryAddressAllocator,CustomDeferredFreeFunctor> ThisType;

        template<class U> using std_get_0 = decltype(std::get<0u>(std::declval<U&>()));
        template<class,class=void> struct is_std_get_0_defined                                   : std::false_type {};
        template<class U> struct is_std_get_0_defined<U,std::void_t<std_get_0<U> > > : std::true_type {};
    protected:
        HeterogenousMemoryAddressAllocator mAllocator;
        ILogicalDevice* mDevice;

        template<typename... Args>
        inline size_type    try_multi_alloc(uint32_t count, size_type* outAddresses, const size_type* bytes, const Args&... args) noexcept
        {
            mAllocator.multi_alloc_addr(count,outAddresses,bytes,args...);

            size_type unallocatedSize = 0;
            for (uint32_t i=0u; i<count; i++)
            {
                if (outAddresses[i]!=invalid_address)
                    continue;

                unallocatedSize += bytes[i];
            }
            return unallocatedSize;
        }

        //! Mutable version for protected usage
        inline HeterogenousMemoryAddressAllocator& getAllocator() noexcept {return mAllocator;}

        inline core::allocator<std::tuple<size_type,size_type> >& getFunctorAllocator() noexcept {return functorAllocator;} // TODO : RobustGeneralpurposeAllocator a-la naughty dog

        class DefaultDeferredFreeFunctor
        {
            private:
                ThisType*   sadbRef;
                size_type*  rangeData;
                size_type   numAllocs;
            public:
				inline DefaultDeferredFreeFunctor(ThisType* _this, size_type numAllocsToFree, const size_type* addrs, const size_type* bytes)
                                                    : sadbRef(_this), rangeData(nullptr), numAllocs(numAllocsToFree)
                {
                    rangeData = reinterpret_cast<size_type*>(sadbRef->getFunctorAllocator().allocate(numAllocs,sizeof(size_type)));
                    memcpy(rangeData            ,addrs,sizeof(size_type)*numAllocs);
                    memcpy(rangeData+numAllocs  ,bytes,sizeof(size_type)*numAllocs);
                }
                DefaultDeferredFreeFunctor(const DefaultDeferredFreeFunctor& other) = delete;
				inline DefaultDeferredFreeFunctor(DefaultDeferredFreeFunctor&& other) : sadbRef(nullptr), rangeData(nullptr), numAllocs(0u)
                {
                    this->operator=(std::forward<DefaultDeferredFreeFunctor>(other));
                }

				inline ~DefaultDeferredFreeFunctor()
                {
                    if (rangeData)
                    {
                        auto alloctr = sadbRef->getFunctorAllocator();
                        alloctr.deallocate(reinterpret_cast<typename std::remove_pointer<decltype(alloctr)>::type::pointer>(rangeData),numAllocs);
                    }
                }

                DefaultDeferredFreeFunctor& operator=(const DefaultDeferredFreeFunctor& other) = delete;
                inline DefaultDeferredFreeFunctor& operator=(DefaultDeferredFreeFunctor&& other)
                {
                    if (rangeData)
                    {
                        auto alloctr = sadbRef->getFunctorAllocator();
                        alloctr.deallocate(reinterpret_cast<typename std::remove_pointer<decltype(alloctr)>::type::pointer>(rangeData),numAllocs);
                    }
                    sadbRef    = other.sadbRef;
                    rangeData   = other.rangeData;
                    numAllocs   = other.numAllocs;
                    other.sadbRef  = nullptr;
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
                    #ifdef _NBL_DEBUG
                    assert(sadbRef && rangeData);
                    #endif // _NBL_DEBUG
                    HeterogenousMemoryAddressAllocator& alloctr = sadbRef->getAllocator();
                    alloctr.multi_free_addr(numAllocs,rangeData,rangeData+numAllocs);
                }
        };
        constexpr static bool UsingDefaultFunctor = std::is_same<CustomDeferredFreeFunctor,void>::value;
        typedef typename std::conditional<UsingDefaultFunctor,DefaultDeferredFreeFunctor,CustomDeferredFreeFunctor>::type DeferredFreeFunctor;
        GPUDeferredEventHandlerST<DeferredFreeFunctor> deferredFrees;
        core::allocator<std::tuple<size_type,size_type> > functorAllocator; // TODO : RobustGeneralpurposeAllocator a-la naughty dog, unbounded allocation, but without resize, use blocks

    public:
        #define DUMMY_DEFAULT_CONSTRUCTOR SubAllocatedDataBuffer() {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR
        //!
        template<typename... Args>
        SubAllocatedDataBuffer(ILogicalDevice* dev, Args&&... args) : mAllocator(std::forward<Args>(args)...), mDevice(dev)
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
        }

        //!
        const HeterogenousMemoryAddressAllocator& getAllocator() const {return mAllocator;}
        //!
        inline const IGPUBuffer*  getBuffer() const noexcept
        {
            auto allocation = mAllocator.getCurrentBufferAllocation();

            IGPUBuffer* retval;
			if constexpr(is_std_get_0_defined<decltype(allocation)>::value)
			{
				retval = std::get<0u>(allocation);
			}
			else
			{
				retval = allocation;
			}
			

            return retval;
        }
        inline IGPUBuffer* getBuffer() noexcept
        {
            return const_cast<IGPUBuffer*>(static_cast<const ThisType*>(this)->getBuffer());
        }

        //! Returns max possible currently allocatable single allocation size, without having to wait for GPU more
        inline size_type    max_size() noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            size_type valueToStopAt = mAllocator.getAddressAllocator().min_size()*3u; // padding, allocation, more padding = 3u
            // we don't actually want or need to poll all possible blocks to free, only first few
            deferredFrees.pollForReadyEvents(valueToStopAt);
            return mAllocator.getAddressAllocator().max_size();
        }
        //! Returns max requestable alignment on the allocation (w.r.t. backing memory start)
        inline size_type    max_alignment() const noexcept {return mAllocator.getAddressAllocator().max_alignment();}


        //!
        template<typename... Args>
        inline size_type    multi_alloc(uint32_t count, Args&&... args) noexcept
        {
            return multi_alloc(GPUEventWrapper::default_wait(),count,std::forward<Args>(args)...);
        }
        //!
        template<class Clock=typename std::chrono::high_resolution_clock, typename... Args>
        inline size_type    multi_alloc(const std::chrono::time_point<Clock>& maxWaitPoint, const Args&... args) noexcept
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
                deferredFrees.waitUntilForReadyEvents(maxWaitPoint,unallocatedSize);

                unallocatedSize = try_multi_alloc(args...);
                if (!unallocatedSize)
                    return 0u;
            } while(Clock::now()<maxWaitPoint);

            return unallocatedSize;
        }
        //!
        inline void         multi_free(core::smart_refctd_ptr<IGPUFence>&& fence, DeferredFreeFunctor&& functor) noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            deferredFrees.addEvent(GPUEventWrapper(mDevice, std::move(fence)),std::forward<DeferredFreeFunctor>(functor));
        }
        inline void         multi_free(uint32_t count, const size_type* addr, const size_type* bytes) noexcept
        {
            #ifdef _NBL_DEBUG
            std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
            assert(tLock.owns_lock());
            #endif // _NBL_DEBUG
            mAllocator.multi_free_addr(count,addr,bytes);
        }
        template<typename Q=DeferredFreeFunctor>
        inline void         multi_free(uint32_t count, const size_type* addr, const size_type* bytes, core::smart_refctd_ptr<IGPUFence>&& fence, typename std::enable_if<std::is_same<Q,DefaultDeferredFreeFunctor>::value>::type* = 0) noexcept
        {
            if (fence)
                multi_free(std::move(fence),DeferredFreeFunctor(this,count,addr,bytes));
            else
                multi_free(count,addr,bytes);
        }
};


template< typename _size_type=uint32_t, class BasicAddressAllocator=core::GeneralpurposeAddressAllocator<_size_type>, class GPUBufferAllocator=SimpleGPUBufferAllocator, class CPUAllocator=core::allocator<uint8_t> >
using SubAllocatedDataBufferST = SubAllocatedDataBuffer<core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,GPUBufferAllocator,CPUAllocator> >;

//MT version?

}

#endif




