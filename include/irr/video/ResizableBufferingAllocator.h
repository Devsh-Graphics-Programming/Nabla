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
        ResizableBufferingAllocatorST(IVideoDriver* inDriver, const CPUAllocator& reservedMemAllocator, size_type bufSz, Args&&... args) :
                                mAllocator(reservedMemAllocator,HostDeviceMirrorBufferAllocator<>(inDriver),bufSz,std::forward<Args>(args)...)
        {
        }

        virtual ~ResizableBufferingAllocatorST() {}


        inline const BasicAddressAllocator& getAddressAllocator() const
        {
            return mAllocator.getAddressAllocator();
        }

        inline void*                        getBackBufferPointer()
        {
            return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getCPUStagingAreaPtr();
        }

        inline IGPUBuffer*                  getFrontBuffer()
        {
            return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getGPUBuffer();
        }


        template<typename... Args>
        inline void                         multi_alloc_addr(Args&&... args) {mAllocator.multi_alloc_addr(std::forward<Args>(args)...);}

        template<typename... Args>
        inline void                         multi_free_addr(Args&&... args) {mAllocator.multi_free_addr(std::forward<Args>(args)...);}


        //! Makes Writes visible, it can fail if there is a lack of space in the streaming buffer to stream with
        template<class StreamingTransientDataBuffer>
        inline bool                         swapBuffers(StreamingTransientDataBuffer* streamingBuff, core::vector<IDriverMemoryAllocation::MappedMemoryRange>& flushRanges)
        {
            uint8_t* data = reinterpret_cast<uint8_t*>(getBackBufferPointer());
            typename StreamingTransientDataBuffer::size_type  dataSize;
            if (Base::alwaysSwapEntireRange)
                dataSize = getFrontBuffer()->getSize();
            else if (Base::dirtyRange.first<Base::dirtyRange.second)
            {
                data += Base::dirtyRange.first;
                dataSize = Base::dirtyRange.second-Base::dirtyRange.first;
            }

            const void* dataPtrWithTypeToMakeForwardingHappy = data;
            typename StreamingTransientDataBuffer::size_type offset = StreamingTransientDataBuffer::invalid_address;
            typename StreamingTransientDataBuffer::size_type alignment = 8u;
            streamingBuff->multi_place(std::chrono::microseconds(1u),1u,&dataPtrWithTypeToMakeForwardingHappy,&offset,&dataSize,&alignment);
            if (offset==StreamingTransientDataBuffer::invalid_address)
                return false;

            auto driver = core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getDriver();
            driver->copyBuffer(streamingBuff->getBuffer(),getFrontBuffer(),offset,0u,dataSize);

            auto fence = driver->placeFence();
            streamingBuff->multi_free(1u,&offset,&dataSize,fence);
            fence->drop();

            Base::resetDirtyRange();
            return true;
        }
};

}
}

#endif // __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__



