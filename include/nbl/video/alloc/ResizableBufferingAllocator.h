// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_RESIZABLE_BUFFERING_ALLOCATOR_H__
#define __NBL_VIDEO_RESIZABLE_BUFFERING_ALLOCATOR_H__

#include "nbl/core/alloc/MultiBufferingAllocatorBase.h"
#include "nbl/core/alloc/ResizableHeterogenousMemoryAllocator.h"
#include "nbl/video/alloc/HostDeviceMirrorBufferAllocator.h"

namespace nbl
{
namespace video
{
template<class BasicAddressAllocator, class CPUAllocator = core::allocator<uint8_t>, bool onlySwapRangesMarkedDirty = false, class CustomDeferredFreeFunctor = void>
class ResizableBufferingAllocatorST : public core::MultiBufferingAllocatorBase<BasicAddressAllocator, onlySwapRangesMarkedDirty>,
                                      protected SubAllocatedDataBuffer<core::ResizableHeterogenousMemoryAllocator<core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator, HostDeviceMirrorBufferAllocator<>, CPUAllocator> >, CustomDeferredFreeFunctor>,
                                      public virtual core::IReferenceCounted
{
    typedef core::MultiBufferingAllocatorBase<BasicAddressAllocator, onlySwapRangesMarkedDirty> MultiBase;
    typedef SubAllocatedDataBuffer<core::ResizableHeterogenousMemoryAllocator<core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator, HostDeviceMirrorBufferAllocator<>, CPUAllocator> > > Base;

protected:
    virtual ~ResizableBufferingAllocatorST() {}

public:
    typedef typename Base::size_type size_type;
    static constexpr size_type invalid_address = Base::invalid_address;

    template<typename... Args>
    ResizableBufferingAllocatorST(IDriver* inDriver, const CPUAllocator& reservedMemAllocator, Args&&... args)
        : Base(reservedMemAllocator, HostDeviceMirrorBufferAllocator<>(inDriver), std::forward<Args>(args)...)
    {
    }

    const auto& getAllocator() const { return Base::getAllocator(); }

    inline const BasicAddressAllocator& getAddressAllocator() const
    {
        return getAllocator().getAddressAllocator();
    }

    inline void* getBackBufferPointer()
    {
        return Base::mAllocator.getCurrentBufferAllocation().second;
    }

    inline IGPUBuffer* getFrontBuffer()
    {
        return Base::getBuffer();
    }

    //! no waiting because of no fencing
    template<typename... Args>
    inline void multi_alloc_addr(Args&&... args) { Base::multi_alloc(std::chrono::steady_clock::time_point(), std::forward<Args>(args)...); }
    //! no fencing of deallocations,the data update is already fenced
    inline void multi_free_addr(uint32_t count, const size_type* addr, const size_type* bytes) { Base::multi_free(count, addr, bytes, nullptr); }

    //! Makes Writes visible, it can fail if there is a lack of space in the streaming buffer to stream with
    template<class StreamingTransientDataBuffer>
    inline bool pushBuffer(StreamingTransientDataBuffer* streamingBuf)
    {
        typename StreamingTransientDataBuffer::size_type dataOffset;
        typename StreamingTransientDataBuffer::size_type dataSize;
        if(MultiBase::alwaysSwapEntireRange)
        {
            dataOffset = 0u;
            dataSize = getFrontBuffer()->getSize();
        }
        else if(MultiBase::pushRange.first < MultiBase::pushRange.second)
        {
            dataOffset = MultiBase::pushRange.first;
            dataSize = MultiBase::pushRange.second - MultiBase::pushRange.first;
        }
        else
            return true;

        auto driver = core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(Base::mAllocator).getDriver();
        driver->updateBufferRangeViaStagingBuffer(getFrontBuffer(), dataOffset, dataSize, reinterpret_cast<uint8_t*>(getBackBufferPointer()) + dataOffset);  // TODO: create and change to non-blocking variant with std::chrono::microseconds(1u)

        MultiBase::resetPushRange();
        return true;
    }
};

}
}

#endif
