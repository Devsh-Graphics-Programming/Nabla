// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_STREAMING_TRANSIENT_DATA_BUFFER_H__
#define __NBL_VIDEO_STREAMING_TRANSIENT_DATA_BUFFER_H__


#include "nbl/core/declarations.h"

#include <cstring>

#include "nbl/video/alloc/SubAllocatedDataBuffer.h"
#include "nbl/video/alloc/StreamingGPUBufferAllocator.h"


namespace nbl::video
{
    
template<typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void, class RecursiveLockable=std::recursive_mutex>
class StreamingTransientDataBufferMT;

template<typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void>
class StreamingTransientDataBufferST : public core::IReferenceCounted
{
        typedef core::HeterogenousMemoryAddressAllocatorAdaptor<core::GeneralpurposeAddressAllocator<_size_type>,StreamingGPUBufferAllocator,CPUAllocator> HeterogenousMemoryAddressAllocator;
        typedef StreamingTransientDataBufferST<_size_type,CPUAllocator> ThisType;
        using Composed = SubAllocatedDataBuffer<HeterogenousMemoryAddressAllocator,CustomDeferredFreeFunctor>;
    protected:
        alignas(Composed) uint8_t m_composedStorage[sizeof(Composed)];
        Composed* getSubAllocatedDataBuffer() {return reinterpret_cast<Composed*>(m_composedStorage);}
        const Composed* getSubAllocatedDataBuffer() const {return reinterpret_cast<const Composed*>(m_composedStorage);}
        
        friend class StreamingTransientDataBufferMT<_size_type,CPUAllocator,CustomDeferredFreeFunctor>;
        virtual ~StreamingTransientDataBufferST()
        {
            getSubAllocatedDataBuffer()->~Composed();
        }
    public:
        using size_type = typename Composed::size_type;
        static constexpr inline size_type invalid_address = Composed::invalid_address;

        // TODO remove
        //StreamingTransientDataBufferST() {}

        //!
        /**
        \param default minAllocSize has been carefully picked to reflect the lowest nonCoherentAtomSize under Vulkan 1.1 which is not 1u .*/
        StreamingTransientDataBufferST(ILogicalDevice* inDevice, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs, const CPUAllocator& reservedMemAllocator=CPUAllocator(), size_type minAllocSize=64u)
        {
            new (getSubAllocatedDataBuffer()) Composed(inDevice,reservedMemAllocator,StreamingGPUBufferAllocator(inDevice,bufferReqs),0u,0u,bufferReqs.vulkanReqs.alignment,bufferReqs.vulkanReqs.size,minAllocSize);
        }

        const auto& getAllocator() const {return getSubAllocatedDataBuffer()->getAllocator();}


        inline bool         needsManualFlushOrInvalidate() const {return !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);}

        inline IGPUBuffer*  getBuffer() noexcept {return getSubAllocatedDataBuffer()->getBuffer();}
        inline const IGPUBuffer*  getBuffer() const noexcept {return getSubAllocatedDataBuffer()->getBuffer();}

        inline void*        getBufferPointer() noexcept {return getSubAllocatedDataBuffer()->getAllocator().getCurrentBufferAllocation().ptr;}

        inline void         cull_frees() noexcept {getSubAllocatedDataBuffer()->cull_frees();}

        inline size_type    max_size() noexcept {return getSubAllocatedDataBuffer()->max_size();}

        inline size_type    max_alignment() const noexcept {return getSubAllocatedDataBuffer()->maxalignment();}


        template<typename... Args>
        inline size_type    multi_place(uint32_t count, Args&&... args) noexcept
        {
            return multi_place(GPUEventWrapper::default_wait(),count,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline size_type    multi_alloc(Args&&... args) noexcept
        {
            return getSubAllocatedDataBuffer()->multi_alloc(std::forward<Args>(args)...);
        }

        template<class Clock=std::chrono::steady_clock, class Duration=typename Clock::duration, typename... Args>
        inline size_type    multi_place(const std::chrono::time_point<Clock,Duration>& maxWaitPoint, uint32_t count, const void* const* dataToPlace, size_type* outAddresses, const size_type* bytes, Args&&... args) noexcept
        {
        #ifdef _NBL_DEBUG
            assert(getBuffer()->getBoundMemory());
        #endif // _NBL_DEBUG
            auto retval = multi_alloc(maxWaitPoint,count,outAddresses,bytes,std::forward<Args>(args)...);
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
            getSubAllocatedDataBuffer()->multi_free(std::forward<Args>(args)...);
        }
};


template<typename _size_type, class CPUAllocator, class CustomDeferredFreeFunctor, class RecursiveLockable>
class StreamingTransientDataBufferMT : public core::IReferenceCounted
{
        using Composed = StreamingTransientDataBufferST<_size_type,CPUAllocator,CustomDeferredFreeFunctor>;
    protected:
        alignas(Composed) uint8_t m_composedStorage[sizeof(Composed)];
        Composed* getSinglethreaded() {return reinterpret_cast<Composed*>(m_composedStorage);}
        const Composed* getSinglethreaded() const {return reinterpret_cast<const Composed*>(m_composedStorage);}
        RecursiveLockable lock;

        virtual ~StreamingTransientDataBufferMT()
        {
            getSinglethreaded()->~Composed();
        }
    public:
        using size_type = typename Composed::size_type;
        static constexpr inline size_type invalid_address = Composed::invalid_address;

        template<typename... Args>
        StreamingTransientDataBufferMT(Args... args)
        {
            new (getSinglethreaded()) Composed(std::forward<Args>(args)...);
        }

        const auto& getAllocator() const {return getSinglethreaded()->getAllocator();}


        inline bool         needsManualFlushOrInvalidate()
        {
            lock.lock();
            bool retval = getSinglethreaded()->needsManualFlushOrInvalidate(); // if this cap doesn't change we can cache it and avoid a stupid lock that protects against invalid buffer pointer
            lock.unlock();
            return retval;
        }


        //! With the right Data Allocator, this pointer should remain constant after first allocation but the underlying gfx API object may change!
        inline IGPUBuffer*  getBuffer() noexcept
        {
            return getSinglethreaded()->getBuffer();
        }

        //! you should really `this->get_lock()`  if you need the pointer to not become invalid while you use it
        inline void*        getBufferPointer() noexcept
        {
            return getSinglethreaded()->getBufferPointer();
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline void    cull_frees() noexcept
        {
            lock.lock();
            getSinglethreaded()->cull_frees();
            lock.unlock();
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline size_type    max_size() noexcept
        {
            lock.lock();
            auto retval = getSinglethreaded()->max_size();
            lock.unlock();
            return retval;
        }


        //! this value should be immutable
        inline size_type    max_alignment() const noexcept {return getSinglethreaded()->maxalignment();}


        template<typename... Args>
        inline size_type    multi_alloc(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = getSinglethreaded()->multi_alloc(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline size_type    multi_place(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = getSinglethreaded()->multi_place(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline void         multi_free(Args&&... args) noexcept
        {
            lock.lock();
            getSinglethreaded()->multi_free(std::forward<Args>(args)...);
            lock.unlock();
        }


        //! Extra == Use WITH EXTREME CAUTION
        inline RecursiveLockable&   get_lock() noexcept
        {
            return lock;
        }
};


}

#endif



