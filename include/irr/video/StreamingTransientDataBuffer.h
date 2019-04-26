#ifndef __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__
#define __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__

#include <cstring>

#include "irr/core/IReferenceCounted.h"
#include "irr/video/SubAllocatedDataBuffer.h"
#include "irr/video/StreamingGPUBufferAllocator.h"
#include "IDriverFence.h"


namespace irr
{
namespace video
{


template< typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void >
class StreamingTransientDataBufferST : protected SubAllocatedDataBuffer<core::HeterogenousMemoryAddressAllocatorAdaptor<core::GeneralpurposeAddressAllocator<_size_type>,StreamingGPUBufferAllocator,CPUAllocator>,CustomDeferredFreeFunctor>,
                                                                public virtual core::IReferenceCounted
{
        typedef core::HeterogenousMemoryAddressAllocatorAdaptor<core::GeneralpurposeAddressAllocator<_size_type>,StreamingGPUBufferAllocator,CPUAllocator> HeterogenousMemoryAddressAllocator;
        typedef StreamingTransientDataBufferST<_size_type,CPUAllocator> ThisType;
        typedef SubAllocatedDataBuffer<HeterogenousMemoryAddressAllocator,CustomDeferredFreeFunctor> Base;
    protected:
        virtual ~StreamingTransientDataBufferST() {}
    public:
        typedef typename Base::size_type    size_type;
        static constexpr size_type                  invalid_address = Base::invalid_address;

        #define DUMMY_DEFAULT_CONSTRUCTOR StreamingTransientDataBufferST() {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR
        //!
        /**
        \param default minAllocSize has been carefully picked to reflect the lowest nonCoherentAtomSize under Vulkan 1.1 which is not 1u .*/
        StreamingTransientDataBufferST(IDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs,
                                       const CPUAllocator& reservedMemAllocator=CPUAllocator(), size_type minAllocSize=64u) :
                                Base(reservedMemAllocator,StreamingGPUBufferAllocator(inDriver,bufferReqs),0u,0u,bufferReqs.vulkanReqs.alignment,bufferReqs.vulkanReqs.size,minAllocSize)
        {
        }

        const auto& getAllocator() const {return Base::getAllocator();}


        inline bool         needsManualFlushOrInvalidate() const {return !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);}

        inline IGPUBuffer*  getBuffer() noexcept {return Base::getBuffer();}
        inline const IGPUBuffer*  getBuffer() const noexcept {return Base::getBuffer();}

        inline void*        getBufferPointer() noexcept {return Base::mAllocator.getCurrentBufferAllocation().second;}


        inline size_type    max_size() noexcept {return Base::max_size();}

        inline size_type    max_alignment() const noexcept {return Base::max_alignment();}


        template<typename... Args>
        inline size_type    multi_place(uint32_t count, Args&&... args) noexcept
        {
            return multi_place(std::chrono::nanoseconds(50000ull),count,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline size_type    multi_alloc(Args&&... args) noexcept
        {
            return Base::multi_alloc(std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline size_type    multi_place(const std::chrono::nanoseconds& maxWait, uint32_t count, const void* const* dataToPlace, size_type* outAddresses, const size_type* bytes, Args&&... args) noexcept
        {
        #ifdef _IRR_DEBUG
            assert(getBuffer()->getBoundMemory());
        #endif // _IRR_DEBUG
            auto retval = multi_alloc(maxWait,count,outAddresses,bytes,std::forward<Args>(args)...);
            // fill with data
            for (uint32_t i=0; i<count; i++)
            {
                if (outAddresses[i]!=invalid_address)
                    memcpy(reinterpret_cast<uint8_t*>(getBufferPointer())+outAddresses[i],dataToPlace[i],bytes[i]);
            }
            return retval;
        }

        template<typename... Args>
        inline void         multi_free(Args&&... args) noexcept
        {
            Base::multi_free(std::forward<Args>(args)...);
        }
};


template< typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void, class RecursiveLockable=std::recursive_mutex>
class StreamingTransientDataBufferMT : protected StreamingTransientDataBufferST<_size_type,CPUAllocator,CustomDeferredFreeFunctor>, public virtual core::IReferenceCounted
{
        typedef StreamingTransientDataBufferST<_size_type,CPUAllocator,CustomDeferredFreeFunctor> Base;
    protected:
        RecursiveLockable lock;

        virtual ~StreamingTransientDataBufferMT() {}
    public:
        typedef typename Base::size_type                        size_type;
        static constexpr size_type                                      invalid_address = Base::invalid_address;

        using Base::Base;

        const auto& getAllocator() const {return Base::getAllocator();}


        inline bool         needsManualFlushOrInvalidate()
        {
            lock.lock();
            bool retval = Base::needsManualFlushOrInvalidate(); // if this cap doesn't change we can cache it and avoid a stupid lock that protects against invalid buffer pointer
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



