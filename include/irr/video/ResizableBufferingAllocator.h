#ifndef __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__
#define __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__


#include "irr/core/alloc/MultiBufferingAllocatorBase.h"
#include "irr/core/alloc/ResizableHeterogenousMemoryAllocator.h"
#include "irr/video/HostDeviceMirrorBufferAllocator.h"

namespace irr
{
namespace video
{

template<class BasicAddressAllocator, class CPUAllocator=core::allocator<uint8_t>, bool onlySwapRangesMarkedDirty = false >
class ResizableBufferingAllocatorST : public core::MultiBufferingAllocatorBase<BasicAddressAllocator,onlySwapRangesMarkedDirty>,
                                       protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor
{
        typedef core::MultiBufferingAllocatorBase<BasicAddressAllocator,onlySwapRangesMarkedDirty>                                                               Base;
    protected:
        typedef core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,HostDeviceMirrorBufferAllocator<>,CPUAllocator> HeterogenousBase;
        core::ResizableHeterogenousMemoryAllocator<HeterogenousBase>                                                                                                            mAllocator;

    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(typename BasicAddressAllocator::size_type);

        template<typename... Args>
        ResizableBufferingAllocatorST(IDriver* inDriver, const CPUAllocator& reservedMemAllocator, Args&&... args) :
                                mAllocator(reservedMemAllocator,HostDeviceMirrorBufferAllocator<>(inDriver),std::forward<Args>(args)...)
        {
        }

        virtual ~ResizableBufferingAllocatorST() {}


        inline const BasicAddressAllocator& getAddressAllocator() const
        {
            return mAllocator.getAddressAllocator();
        }

        inline void*                        getBackBufferPointer()
        {
            return mAllocator.getCurrentBufferAllocation().second;
        }

        inline IGPUBuffer*                  getFrontBuffer()
        {
            return mAllocator.getCurrentBufferAllocation().first;
        }


        template<typename... Args>
        inline void                         multi_alloc_addr(Args&&... args) {mAllocator.multi_alloc_addr(std::forward<Args>(args)...);}

        template<typename... Args>
        inline void                         multi_free_addr(Args&&... args) {mAllocator.multi_free_addr(std::forward<Args>(args)...);}


        //! Makes Writes visible, it can fail if there is a lack of space in the streaming buffer to stream with
        template<class StreamingTransientDataBuffer>
        inline bool                         swapBuffers(StreamingTransientDataBuffer* streamingBuff)
        {
            typename StreamingTransientDataBuffer::size_type  dataOffset;
            typename StreamingTransientDataBuffer::size_type  dataSize;
            if (Base::alwaysSwapEntireRange)
            {
                dataOffset = 0u;
                dataSize = getFrontBuffer()->getSize();
            }
            else if (Base::dirtyRange.first<Base::dirtyRange.second)
            {
                dataOffset = Base::dirtyRange.first;
                dataSize = Base::dirtyRange.second-Base::dirtyRange.first;
            }
            else
                return true;

            auto driver = core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getDriver();
            driver->updateBufferRangeViaStagingBuffer(getFrontBuffer(),dataOffset,dataSize,reinterpret_cast<uint8_t*>(getBackBufferPointer())+dataOffset); // TODO: create and change to non-blocking variant with std::chrono::microseconds(1u)

            Base::resetDirtyRange();
            return true;
        }
};

}
}

#endif // __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__



