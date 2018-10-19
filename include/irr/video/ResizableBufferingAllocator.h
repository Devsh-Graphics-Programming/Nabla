#ifndef __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__
#define __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__


#include "irr/core/alloc/MultiBufferingAllocatorBase.h"
#include "irr/core/alloc/ResizableHeterogenousMemoryAllocator.h"
#include "irr/video/SimpleGPUBufferAllocator.h"

namespace irr
{
namespace video
{

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
class HostDeviceMirrorBufferAllocator : public GPUMemoryAllocatorBase
{
        std::pair<void*,IGPUBuffer*>                 lastAllocation;

        inline auto         createBuffers(size_t bytes, size_t alignment=_IRR_SIMD_ALIGNMENT)
        {
            decltype(lastAllocation) retval;
            retval.first = _IRR_ALIGNED_MALLOC(bytes,alignment);

            auto memReqs = mDriver->getDeviceLocalGPUMemoryReqs();
            memReqs.vulkanReqs.size = bytes;
            retval.second = mDriver->createGPUBufferOnDedMem(memReqs,false);

            return retval;
        }
    public:
        HostDeviceMirrorBufferAllocator(IVideoDriver* inDriver) : GPUMemoryAllocatorBase(inDriver), lastAllocation(nullptr,nullptr) {}

        inline void*        allocate(size_t bytes) noexcept
        {
        #ifdef _DEBUG
            assert(!lastAllocation.first && !lastAllocation.second);
        #endif // _DEBUG
            lastAllocation = createBuffers(bytes);
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets) noexcept
        {
        #ifdef _DEBUG
            assert(!addr && lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG

            auto alignment = allocToQueryOffsets.max_alignment();
            auto oldMemReqs = lastAllocation.second->getMemoryReqs();
            auto tmp = createBuffers(bytes,alignment);

            //move contents
            size_t oldOffset = allocToQueryOffsets.get_align_offset();
            auto copyRangeLen = std::min(oldMemReqs.vulkanReqs.size-oldOffset,bytes);

            memcpy(tmp.first,lastAllocation.first+oldOffset,copyRangeLen);
            _IRR_ALIGNED_FREE(lastAllocation.first);
            mDriver->copyBuffer(lastAllocation.second,tmp.second,oldOffset,0u,copyRangeLen);
            //swap the internals of buffers
            lastAllocation.second->pseudoMoveAssign(tmp.second);
            tmp.second->drop();

            //book-keeping and return
            lastAllocation.first = tmp.first;
            return lastAllocation.first;
        }

        inline void         deallocate(void* addr) noexcept
        {
        #ifdef _DEBUG
            assert(!addr && lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG
            _IRR_ALIGNED_FREE(lastAllocation.first);
            lastAllocation.second->drop();
            lastAllocation.first = nullptr;
            lastAllocation.second = nullptr;
        }


        // extras
        inline void*        getCPUStagingAreaPtr()
        {
            return lastAllocation.first;
        }

        inline IGPUBuffer*  getGPUBuffer()
        {
            return lastAllocation.second;
        }
};

template<class BasicAddressAllocator, class CPUAllocator=core::allocator<uint8_t>, bool onlySwapRangesMarkedDirty = false >
class ResizableBufferingAllocatorST : public core::MultiBufferingAllocatorBase<BasicAddressAllocator,onlySwapRangesMarkedDirty>,
                                       protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor
{
    protected:
        ResizableHeterogenousMemoryAllocator<BasicAddressAllocator,HostDeviceMirrorBufferAllocator,CPUAllocator>    mAllocator;
        typedef core::address_allocator_traits<decltype(mAllocator)>                                                alloc_traits;

    public:
        typedef typename BasicAddressAllocator::size_type                                                           size_type;

        template<typename... Args>
        ResizableBufferingAllocatorST(IVideoDriver* inDriver, const CPUAllocator& reservedMemAllocator, size_type bufSz, Args&&... args) :
                                mAllocator(reservedMemAllocator,HostDeviceMirrorBufferAllocator(inDriver),bufSz,std::forward<Args>(args)...)
        {
        }

        virtual ~ResizableBufferingAllocatorST() {}


        inline const BasicAddressAllocator& getAddressAllocator() const
        {
            return core::impl::FriendOfResizableAddressAllocatorAdaptor::getAddressAllocator(mAllocator);
        }

        inline void*                        getBackBufferPointer()
        {
            return core::impl::FriendOfResizableAddressAllocatorAdaptor::getDataAllocator(mAllocator).getCPUStagingAreaPtr();
        }

        inline IGPUBuffer*                  getFrontBuffer()
        {
            return core::impl::FriendOfResizableAddressAllocatorAdaptor::getDataAllocator(mAllocator).getGPUBuffer();
        }


        template<typename... Args>
        inline void                         multi_alloc_addr(Args&&... args) {alloc_traits::multi_alloc_addr(mAllocator,std::forward<Args>(args)...);}

        template<typename... Args>
        inline void                         multi_free_addr(Args&&... args) {alloc_traits::multi_free_addr(mAllocator,std::forward<Args>(args)...);}


        //! Makes Writes visible, it can fail if there is a lack of space in the streaming buffer to stream with
        template<class StreamingTransientDataBuffer>
        inline bool                         swapBuffers(StreamingTransientDataBuffer* streamingBuff, core::vector<IDriverMemoryAllocation::MappedMemoryRange>& flushRanges)
        {
            uint8_t* data = getBackBufferPointer();
            typename AddressAllocator::size_type dataSize;
            if (Base::alwaysSwapEntireRange)
                dataSize = core::FriendOfResizableAddressAllocatorAdaptor::getDataBufferSize();
            else if (Base::dirtyRange.first<Base::dirtyRange.second)
            {
                data += Base::dirtyRange.first;
                dataSize = Base::dirtyRange.second-Base::dirtyRange.first;
            }

            auto offset = streamingBuff->Place(flushRanges,data,dataSize,1u); //! TODO: Make a keep-trying-to-allocate mode in streamingBuff
            if (offset==StreamingTransientDataBuffer::invalid_address)
                return false;

            mDriver->copyBuffer(streamingBuff->getBuffer(),getFrontBuffer(),offset,0u,dataSize);
            auto fence = Driver->placeFence();
            streamingBuff->Free(offset,dataSize,fence);
            fence->drop();
            Base::resetDirtyRange();
            return true;
        }
};

}
}

#endif // __IRR_RESIZABLE_BUFFERING_ALLOCATOR_H__



