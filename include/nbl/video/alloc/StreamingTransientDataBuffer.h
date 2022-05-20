// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_STREAMING_TRANSIENT_DATA_BUFFER_H_
#define _NBL_VIDEO_STREAMING_TRANSIENT_DATA_BUFFER_H_


#include "nbl/core/declarations.h"

#include <cstring>

#include "nbl/video/alloc/CAsyncSingleBufferSubAllocator.h"


namespace nbl::video
{
    
template<class HostAllocator=core::allocator<uint8_t>, class RecursiveLockable=std::recursive_mutex>
class StreamingTransientDataBufferMT;

namespace impl
{
template<class HostAllocator>
class StreamingTransientDataBuffer
{
        using ThisType = StreamingTransientDataBuffer<HostAllocator>;
        using Composed = impl::CAsyncSingleBufferSubAllocator<core::GeneralpurposeAddressAllocator<uint32_t>,HostAllocator>;

    protected:
        virtual ~StreamingTransientDataBuffer() {}

        Composed m_composed;

    public:
        using size_type = typename Composed::size_type;
        static constexpr inline size_type invalid_value = Composed::invalid_value;

#if 0
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
#endif

        //
        inline bool         needsManualFlushOrInvalidate() const {return getBuffer()->getBoundMemory()->haveToMakeVisible();}

        // getters
        inline IGPUBuffer*  getBuffer() noexcept {return m_composed.getBuffer();}
        inline const IGPUBuffer*  getBuffer() const noexcept {return m_composed.getBuffer();}
#if 0
        inline void*        getBufferPointer() noexcept {return m_composed.getAllocator().getCurrentBufferAllocation().ptr;}
#endif
        //
        inline void         cull_frees() noexcept {m_composed.cull_frees();}

        //
        inline size_type    max_size() noexcept {return m_composed.max_size();}

#if 0
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
#endif
};
}

template<class HostAllocator=core::allocator<uint8_t>>
class StreamingTransientDataBufferST : public core::IReferenceCounted, public impl::StreamingTransientDataBuffer<HostAllocator>
{
        using Base = impl::StreamingTransientDataBuffer<HostAllocator>;

    protected:
        ~StreamingTransientDataBufferST() = default;

    public:
        template<typename... Args>
        StreamingTransientDataBufferST(Args&&... args) : Base(std::forward<Args>(args)...) {}
};

template<class HostAllocator, class RecursiveLockable>
class StreamingTransientDataBufferMT : public core::IReferenceCounted
{
        using Composed = impl::StreamingTransientDataBuffer<HostAllocator>;

    protected:
        Composed m_composed;
        RecursiveLockable lock;

        virtual ~StreamingTransientDataBufferMT() {}

    public:
        using size_type = typename Composed::size_type;
        static constexpr inline size_type invalid_value = Composed::invalid_value;

        template<typename... Args>
        StreamingTransientDataBufferMT(Args... args) : m_composed(std::forward<Args>(args)...) {}
#if 0
        //
        inline bool         needsManualFlushOrInvalidate()
        {
            return m_composed.needsManualFlushOrInvalidate();
        }
#endif

        //! With the right Data Allocator, this pointer should remain constant after first allocation but the underlying gfx API object may change!
        inline IGPUBuffer*  getBuffer() noexcept
        {
            return m_composed.getBuffer();
        }
#if 0
        //! you should really `this->get_lock()`  if you need the pointer to not become invalid while you use it
        inline void*        getBufferPointer() noexcept
        {
            return m_composed.getBufferPointer();
        }
#endif
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

#if 0
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
#endif

        //! Extra == Use WITH EXTREME CAUTION
        inline RecursiveLockable&   get_lock() noexcept
        {
            return lock;
        }
};


}

#endif



