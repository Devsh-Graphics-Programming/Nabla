#ifndef __IRR_SUB_ALLOCATED_DATA_BUFFER_H__
#define __IRR_SUB_ALLOCATED_DATA_BUFFER_H__


#include "irr/core/IReferenceCounted.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"
#include "irr/video/SimpleGPUBufferAllocator.h"

namespace irr
{
namespace video
{

// this buffer is not growable
template< typename _size_type=uint32_t, class GPUBufferAllocator=SimpleGPUBufferAllocator, class CPUAllocator=core::allocator<uint8_t> >
class SubAllocatedDataBufferST : public virtual core::IReferenceCounted
{
    protected:
        typedef core::GeneralpurposeAddressAllocator<_size_type>                                                                                             BasicAddressAllocator;
        typedef core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,GPUBufferAllocator,CPUAllocator> AddressAllocator;
        AddressAllocator mAllocator;
    public:
        typedef typename BasicAddressAllocator::size_type   size_type;
        static constexpr size_type                                            invalid_address = BasicAddressAllocator::invalid_address;

        #define DUMMY_DEFAULT_CONSTRUCTOR SubAllocatedDataBufferST() {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR
        //!
        template<typename... Args>
        SubAllocatedDataBufferST(size_type bufferSize, size_type maxAllocatableAlignment, const GPUBufferAllocator& deviceAllocator,
                                       const CPUAllocator& reservedMemAllocator=CPUAllocator(), Args&&... args) :
                                mAllocator(reservedMemAllocator,deviceAllocator,bufferSize,maxAllocatableAlignment,std::forward<Args>(args)...)
        {
        }

        virtual ~SubAllocatedDataBufferST() {}

        const AddressAllocator& getAllocator() const {return mAllocator;}

/*
        inline IGPUBuffer*  getBuffer() noexcept {return mAllocator.getCurrentBufferAllocation().first;}


        inline size_type    max_size() noexcept
        {
            size_type valueToStopAt = mAllocator.getAddressAllocator().min_size()*3u; // padding, allocation, more padding = 3u
            deferredFrees.pollForReadyEvents(valueToStopAt);
            return mAllocator.getAddressAllocator().max_size();
        }


        inline size_type    max_alignment() const noexcept {return mAllocator.getAddressAllocator().max_alignment();}


        template<typename... Args>
        inline size_type    multi_alloc(uint32_t count, size_type* outAddresses, const size_type* bytes, Args&&... args) noexcept
        {
            return multi_alloc(std::chrono::nanoseconds(50000ull),count,outAddresses,bytes,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline size_type    multi_alloc(const std::chrono::nanoseconds& maxWait, uint32_t count, size_type* outAddresses, const size_type* bytes, const Args&... args) noexcept
        {
            // try allocate once
            size_type unallocatedSize = try_multi_alloc(count,outAddresses,bytes,args...);
            if (!unallocatedSize)
                return 0u;

            auto maxWaitPoint = std::chrono::high_resolution_clock::now()+maxWait; // 50 us
            // then try to wait at least once and allocate
            do
            {
                deferredFrees.waitUntilForReadyEvents(maxWaitPoint,unallocatedSize);

                unallocatedSize = try_multi_alloc(count,outAddresses,bytes,args...);
                if (!unallocatedSize)
                    return 0u;
            } while(std::chrono::high_resolution_clock::now()<maxWaitPoint);

            return unallocatedSize;
        }

        inline void         multi_free(uint32_t count, const size_type* addr, const size_type* bytes, IDriverFence* fence) noexcept
        {
            if (fence)
                deferredFrees.addEvent(GPUEventWrapper(fence),DeferredFreeFunctor(&mAllocator,count,addr,bytes));
            else
                mAllocator.multi_free_addr(count,addr,bytes);
        }*/
    protected:/*
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

        class DeferredFreeFunctor : protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor
        {
            public:
                DeferredFreeFunctor(core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,GPUBufferAllocator,CPUAllocator>* alloctr,
                                    size_type numAllocsToFree, const size_type* addrs, const size_type* bytes) : allocRef(alloctr), rangeData(nullptr), numAllocs(numAllocsToFree)
                {
                    rangeData = reinterpret_cast<size_type*>(getHostAllocator(*allocRef).allocate(sizeof(size_type)*numAllocs*2u,sizeof(size_type))); // TODO : RobustPoolAllocator
                    memcpy(rangeData            ,addrs,sizeof(size_type)*numAllocs);
                    memcpy(rangeData+numAllocs  ,bytes,sizeof(size_type)*numAllocs);
                }
                DeferredFreeFunctor(const DeferredFreeFunctor& other) = delete;
                DeferredFreeFunctor(DeferredFreeFunctor&& other) : allocRef(nullptr), rangeData(nullptr), numAllocs(0u)
                {
                    this->operator=(std::forward<DeferredFreeFunctor>(other));
                }

                ~DeferredFreeFunctor()
                {
                    if (rangeData)
                        getHostAllocator(*allocRef).deallocate(reinterpret_cast<typename CPUAllocator::pointer>(rangeData),sizeof(size_type)*numAllocs*2u);// TODO : RobustPoolAllocator
                }

                DeferredFreeFunctor& operator=(const DeferredFreeFunctor& other) = delete;
                inline DeferredFreeFunctor& operator=(DeferredFreeFunctor&& other)
                {
                    if (rangeData)
                        getHostAllocator(*allocRef).deallocate(reinterpret_cast<typename CPUAllocator::pointer>(rangeData),sizeof(size_type)*numAllocs*2u);// TODO : RobustPoolAllocator
                    allocRef    = other.allocRef;
                    rangeData   = other.rangeData;
                    numAllocs   = other.numAllocs;
                    other.allocRef  = nullptr;
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
                    #ifdef _DEBUG
                    assert(allocRef && rangeData);
                    #endif // _DEBUG
                    allocRef->multi_free_addr(numAllocs,rangeData,rangeData+numAllocs);
                }

            private:
                core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,GPUBufferAllocator,CPUAllocator>*    allocRef;
                size_type*                                                                                                          rangeData; // TODO : RobustPoolAllocator
                size_type                                                                                                           numAllocs;
        };
        GPUEventDeferredHandlerST<DeferredFreeFunctor> deferredFrees;*/
};

//MT version?

}
}

#endif // __IRR_SUB_ALLOCATED_DATA_BUFFER_H__




