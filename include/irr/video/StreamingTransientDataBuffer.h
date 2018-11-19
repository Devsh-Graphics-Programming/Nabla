#ifndef __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__
#define __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__


#include "irr/core/IReferenceCounted.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"
#include "irr/video/StreamingGPUBufferAllocator.h"

namespace irr
{
namespace video
{


template< typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t> >
class StreamingTransientDataBufferST : protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor, public virtual core::IReferenceCounted
{
    protected:
        typedef core::GeneralpurposeAddressAllocator<_size_type>                                                        BasicAddressAllocator;
        core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator> mAllocator; // no point for a streaming buffer to grow
    public:
        typedef typename BasicAddressAllocator::size_type           size_type;
        static constexpr size_type                                  invalid_address = BasicAddressAllocator::invalid_address;

        #define DUMMY_DEFAULT_CONSTRUCTOR StreamingTransientDataBufferST() {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR
        //!
        /**
        \param default minAllocSize has been carefully picked to reflect the lowest nonCoherentAtomSize under Vulkan 1.1 which is not 1u .*/
        StreamingTransientDataBufferST(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs,
                                       const CPUAllocator& reservedMemAllocator=CPUAllocator(), size_type minAllocSize=64u) :
                                mAllocator(reservedMemAllocator,StreamingGPUBufferAllocator(inDriver,bufferReqs),bufferReqs.vulkanReqs.size,minAllocSize)
        {
        }

        template<typename... Args>
        StreamingTransientDataBufferST(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs,
                                       const CPUAllocator& reservedMemAllocator=CPUAllocator(), Args&&... args) :
                                mAllocator(reservedMemAllocator,StreamingGPUBufferAllocator(inDriver,bufferReqs),bufferReqs.vulkanReqs.size,std::forward<Args>(args)...)
        {
        }

        virtual ~StreamingTransientDataBufferST() {}


        inline bool         needsManualFlushOrInvalidate() const {return !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);}

        inline IGPUBuffer*  getBuffer() noexcept {return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getAllocatedBuffer();}

        inline void*        getBufferPointer() noexcept {return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getAllocatedPointer();}


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
        inline size_type    multi_place(uint32_t count, const void* const* dataToPlace, size_type* outAddresses, const size_type* bytes, Args&&... args) noexcept
        {
            return multi_place(std::chrono::nanoseconds(50000ull),count,dataToPlace,outAddresses,bytes,std::forward<Args>(args)...);
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

        template<typename... Args>
        inline size_type    multi_place(const std::chrono::nanoseconds& maxWait, uint32_t count, const void* const* dataToPlace, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args) noexcept
        {
        #ifdef _DEBUG
            assert(getBuffer()->getBoundMemory());
        #endif // _DEBUG
            auto retval = multi_alloc(maxWait,count,outAddresses,bytes,alignment,args...);
            // fill with data
            for (uint32_t i=0; i<count; i++)
            {
                if (outAddresses[i]!=invalid_address)
                    memcpy(reinterpret_cast<uint8_t*>(getBufferPointer())+outAddresses[i],dataToPlace[i],bytes[i]);
            }
            return retval;
        }

        inline void         multi_free(uint32_t count, const size_type* addr, const size_type* bytes, IDriverFence* fence) noexcept
        {
            if (fence)
                deferredFrees.addEvent(GPUEventWrapper(fence),DeferredFreeFunctor(&mAllocator,count,addr,bytes));
            else
                mAllocator.multi_free_addr(count,addr,bytes);
        }
    protected:
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
                DeferredFreeFunctor(core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator>* alloctr,
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
                core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator>*    allocRef;
                size_type*                                                                                                          rangeData; // TODO : RobustPoolAllocator
                size_type                                                                                                           numAllocs;
        };
        GPUEventDeferredHandlerST<DeferredFreeFunctor> deferredFrees;
};


template< typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class RecursiveLockable=std::recursive_mutex>
class StreamingTransientDataBufferMT : protected StreamingTransientDataBufferST<_size_type,CPUAllocator>, public virtual core::IReferenceCounted
{
        typedef StreamingTransientDataBufferST<_size_type,CPUAllocator> Base;
    protected:
        RecursiveLockable lock;
    public:
        typedef typename Base::size_type                        size_type;
        static constexpr size_type                                      invalid_address = Base::invalid_address;

        using Base::Base;


        inline bool         needsManualFlushOrInvalidate()
        {
            lock.lock();
            bool retval = !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);
            lock.unlock();
            return retval;
        }


        //! With the right Data Allocator, this pointer should remain constant after first allocation but the underlying gfx API object may change!
        inline IGPUBuffer*  getBuffer() noexcept
        {
            return Base::getBuffer();
        }

        //! you should really `this->get_lock()`  if you need the pointer to not become invalid while you use it
        inline void*        getBufferPointer() noexcept
        {
            return Base::getBufferPointer();
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline size_type    max_size() noexcept
        {
            lock.lock();
            auto retval = Base::max_size();
            lock.unlock();
            return retval;
        }


        //! this value should be immutable
        inline size_type    max_alignment() const noexcept {return Base::max_alignment();}


        template<typename... Args>
        inline size_type    multi_alloc(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = Base::multi_alloc(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline size_type    multi_place(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = Base::multi_place(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline void         multi_free(Args&&... args) noexcept
        {
            lock.lock();
            Base::multi_free(std::forward<Args>(args)...);
            lock.unlock();
        }


        //! Extra == Use WITH EXTREME CAUTION
        inline RecursiveLockable&   get_lock() noexcept
        {
            return lock;
        }
};


}
}

#endif // __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__



