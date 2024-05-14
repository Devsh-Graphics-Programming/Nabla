// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
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
        Composed m_composed;

    public:
        using size_type = typename Composed::size_type;
        using value_type = typename Composed::value_type;
        static constexpr inline size_type invalid_value = Composed::invalid_value;

        // perfect forward ctor to `CAsyncSingleBufferSubAllocator`
        template<typename... Args>
        inline StreamingTransientDataBuffer(asset::SBufferRange<IGPUBuffer>&& _bufferRange, Args&&... args) : m_composed(std::move(_bufferRange),std::forward<Args>(args)...)
        {
            assert(getBuffer()->getBoundMemory().memory->isMappable());
            assert(getBuffer()->getBoundMemory().memory->getMappedPointer());
            // we're suballocating from a buffer, whole buffer needs to be reachable from the mapped pointer
            const auto mappedRange = getBuffer()->getBoundMemory().memory->getMappedRange();
            assert(mappedRange.offset<=getBuffer()->getBoundMemory().offset);
            assert(mappedRange.offset+mappedRange.length>=getBuffer()->getBoundMemory().offset+getBuffer()->getSize());
        }
        virtual ~StreamingTransientDataBuffer() {}

        //
        inline bool needsManualFlushOrInvalidate() const {return getBuffer()->getBoundMemory().memory->haveToMakeVisible();}

        // getters
        inline IGPUBuffer* getBuffer() noexcept {return m_composed.getBuffer();}
        inline const IGPUBuffer* getBuffer() const noexcept {return m_composed.getBuffer();}

        //
        inline void* getBufferPointer() noexcept
        {
            const auto bound = getBuffer()->getBoundMemory();
            return reinterpret_cast<uint8_t*>(bound.memory->getMappedPointer())+bound.offset;
        }

        //
        inline size_type get_total_size() const noexcept {return m_composed.getAddressAllocator().get_total_size();}

        //
        inline uint32_t cull_frees() noexcept {return m_composed.cull_frees();}

        //
        inline size_type max_size() noexcept {return m_composed.max_size();}

        // perfect forward to `Composed` method
        template<typename... Args>
        inline value_type multi_allocate(Args&&... args) noexcept
        {
            return m_composed.multi_allocate(std::forward<Args>(args)...);
        }

        //
        template<typename... Args>
        inline void multi_deallocate(Args&&... args) noexcept
        {
            m_composed.multi_deallocate(std::forward<Args>(args)...);
        }

        // allocate and copy data into the allocations, specifying a timeout for the the allocation
        template<class Clock=std::chrono::steady_clock, class Duration=typename Clock::duration, typename... Args>
        inline size_type multi_place(
            const std::chrono::time_point<Clock,Duration>& maxWaitPoint,
            uint32_t count, const void* const* dataToPlace,
            value_type* outAddresses, const size_type* byteSizes, Args&&... args
        ) noexcept
        {
            auto retval = multi_alloc(maxWaitPoint,count,outAddresses,byteSizes,std::forward<Args>(args)...);
            // fill with data
            for (uint32_t i=0; i<count; i++)
            {
                if (outAddresses[i]!=invalid_value)
                    memcpy(reinterpret_cast<uint8_t*>(getBufferPointer())+outAddresses[i],dataToPlace[i],byteSizes[i]);
            }
            return retval;
        }
        // overload with default timeout
        template<typename... Args>
        inline size_type multi_place(uint32_t count, Args&&... args) noexcept
        {
            return multi_place(TimelineEventHandlerBase::default_wait(),count,std::forward<Args>(args)...);
        }
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

        //
        inline bool needsManualFlushOrInvalidate() const
        {
            return m_composed.needsManualFlushOrInvalidate();
        }

        //
        inline size_type get_total_size() const noexcept
        {
            return m_composed.get_total_size();
        }

        //
        inline IGPUBuffer* getBuffer() noexcept
        {
            return m_composed.getBuffer();
        }
        inline const IGPUBuffer* getBuffer() const noexcept
        {
            return m_composed.getBuffer();
        }

        //
        inline void* getBufferPointer() noexcept
        {
            return m_composed.getBufferPointer();
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline uint32_t cull_frees() noexcept
        {
            lock.lock();
            auto retval = m_composed.cull_frees();
            lock.unlock();
            return retval;
        }

        //! you should really `this->get_lock()` if you need the guarantee that you'll be able to allocate a block of this size!
        inline size_type max_size() noexcept
        {
            lock.lock();
            auto retval = m_composed.max_size();
            lock.unlock();
            return retval;
        }


        template<typename... Args>
        inline size_type multi_allocate(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = m_composed.multi_allocate(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }
        template<typename... Args>
        inline void multi_deallocate(Args&&... args) noexcept
        {
            lock.lock();
            m_composed.multi_deallocate(std::forward<Args>(args)...);
            lock.unlock();
        }
        template<typename... Args>
        inline size_type multi_place(Args&&... args) noexcept
        {
            lock.lock();
            auto retval = m_composed.multi_place(std::forward<Args>(args)...);
            lock.unlock();
            return retval;
        }

        //! Extra == Use WITH EXTREME CAUTION
        inline RecursiveLockable& get_lock() noexcept
        {
            return lock;
        }
};

}

#endif



