// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_STREAMING_TRANSIENT_DATA_BUFFER_H__
#define __NBL_VIDEO_STREAMING_TRANSIENT_DATA_BUFFER_H__


#include "nbl/core/declarations.h"

#include <cstring>

#include "nbl/video/alloc/SubAllocatedDataBuffer.h"
#include "nbl/video/alloc/CStreamingBufferAllocator.h"


namespace nbl::video
{
    
template<typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void, class RecursiveLockable=std::recursive_mutex>
class StreamingTransientDataBufferMT;

namespace impl
{
template<typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void>
class StreamingTransientDataBuffer
{
        typedef core::HeterogenousMemoryAddressAllocatorAdaptor<core::GeneralpurposeAddressAllocator<_size_type>,CStreamingBufferAllocator,CPUAllocator> HeterogenousMemoryAddressAllocator;
        typedef StreamingTransientDataBuffer<_size_type,CPUAllocator> ThisType;
        using Composed = impl::SubAllocatedDataBuffer<HeterogenousMemoryAddressAllocator,CustomDeferredFreeFunctor>;
    protected:
        Composed m_composed;
    public:
        using size_type = typename Composed::size_type;
        static constexpr inline size_type invalid_address = Composed::invalid_address;
        
        virtual ~StreamingTransientDataBuffer() {}

        //!
        /**
        \param default minAllocSize has been carefully picked to reflect the lowest nonCoherentAtomSize under Vulkan 1.1 which is not 1u .*/
        StreamingTransientDataBuffer(
            ILogicalDevice* inDevice,
            const uint32_t usableMemoryTypeBits,
            const IGPUBuffer::SCreationParams& bufferCreationParams,
            const core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
            const CPUAllocator& reservedMemAllocator = CPUAllocator(), size_type minAllocSize = 1024u)
            : m_composed(inDevice, reservedMemAllocator, CStreamingBufferAllocator(inDevice, usableMemoryTypeBits), 0u, 0u, bufferCreationParams, allocateFlags, minAllocSize) {}

        const auto& getAllocator() const {return m_composed.getAllocator();}


        // TODO(Erfan): Replace with getBuffer()->getBoundMemory()->getMemoryTypeFlags or something and assert it has Coherent bit + cache memoryTypeFlags in IDriverMemoryAllocation
        inline bool         needsManualFlushOrInvalidate() const {return !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);}

        inline IGPUBuffer*  getBuffer() noexcept {return m_composed.getBuffer();}
        inline const IGPUBuffer*  getBuffer() const noexcept {return m_composed.getBuffer();}

        inline void*        getBufferPointer() noexcept {return m_composed.getAllocator().getCurrentBufferAllocation().ptr;}

        inline void         cull_frees() noexcept {m_composed.cull_frees();}

        inline size_type    max_size() noexcept {return m_composed.max_size();}

        inline size_type    max_alignment() const noexcept {return m_composed.maxalignment();}


        template<typename... Args>
        inline size_type    multi_place(uint32_t count, Args&&... args) noexcept
        {
            return multi_place(GPUEventWrapper::default_wait(),count,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline size_type    multi_alloc(Args&&... args) noexcept
        {
            return m_composed.multi_alloc(std::forward<Args>(args)...);
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
            m_composed.multi_free(std::forward<Args>(args)...);
        }
};
}

template<typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t>, class CustomDeferredFreeFunctor=void>
class StreamingTransientDataBufferST : public core::IReferenceCounted, public impl::StreamingTransientDataBuffer<_size_type,CPUAllocator,CustomDeferredFreeFunctor>
{
        using Base = impl::StreamingTransientDataBuffer<_size_type,CPUAllocator,CustomDeferredFreeFunctor>;
    protected:
        ~StreamingTransientDataBufferST() = default;
    public:
        template<typename... Args>
        StreamingTransientDataBufferST(Args&&... args) : Base(std::forward<Args>(args)...) {}
};

template<typename _size_type, class CPUAllocator, class CustomDeferredFreeFunctor, class RecursiveLockable>
class StreamingTransientDataBufferMT : public core::IReferenceCounted
{
        using Composed = impl::StreamingTransientDataBuffer<_size_type,CPUAllocator,CustomDeferredFreeFunctor>;
    protected:
        Composed m_composed;
        RecursiveLockable lock;

        virtual ~StreamingTransientDataBufferMT() {}
    public:
        using size_type = typename Composed::size_type;
        static constexpr inline size_type invalid_address = Composed::invalid_address;

        template<typename... Args>
        StreamingTransientDataBufferMT(Args... args) : m_composed(std::forward<Args>(args)...) {}

        const auto& getAllocator() const {return m_composed.getAllocator();}


        inline bool         needsManualFlushOrInvalidate()
        {
            lock.lock();
            bool retval = m_composed.needsManualFlushOrInvalidate(); // if this cap doesn't change we can cache it and avoid a stupid lock that protects against invalid buffer pointer
            lock.unlock();
            return retval;
        }


        //! With the right Data Allocator, this pointer should remain constant after first allocation but the underlying gfx API object may change!
        inline IGPUBuffer*  getBuffer() noexcept
        {
            return m_composed.getBuffer();
        }

        //! you should really `this->get_lock()`  if you need the pointer to not become invalid while you use it
        inline void*        getBufferPointer() noexcept
        {
            return m_composed.getBufferPointer();
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline void    cull_frees() noexcept
        {
            lock.lock();
            m_composed.cull_frees();
            lock.unlock();
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline size_type    max_size() noexcept
        {
            lock.lock();
            auto retval = m_composed.max_size();
            lock.unlock();
            return retval;
        }


        //! this value should be immutable
        inline size_type    max_alignment() const noexcept {return m_composed.maxalignment();}


        template<typename... Args>
        inline size_type    multi_alloc(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = m_composed.multi_alloc(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline size_type    multi_place(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = m_composed.multi_place(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        template<typename... Args>
        inline void         multi_free(Args&&... args) noexcept
        {
            lock.lock();
            m_composed.multi_free(std::forward<Args>(args)...);
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



