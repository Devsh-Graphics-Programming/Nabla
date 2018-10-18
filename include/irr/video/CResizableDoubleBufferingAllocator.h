#ifndef __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__
#define __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__



#include "irr/video/CDoubleBufferingAllocator.h"

namespace irr
{
namespace video
{

class GPUMemoryAllocatorBase
{
    protected:
        IVideoDriver* mDriver;
        IDriverMemoryBacked::SDriverMemoryRequirements mBufferMemReqs;
    public:
        #define DUMMY_DEFAULT_CONSTRUCTOR GPUMemoryAllocatorBase() : mDriver(nullptr) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        GPUMemoryAllocatorBase(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) :
                                                                    mDriver(inDriver), mBufferMemReqs(bufferReqs) {}

        virtual ~GPUMemoryAllocatorBase() {}

        inline auto     getCurrentMemReqs() const {return mBufferMemReqs;}

        inline size_t   min_alignment() const
        {
            return mDriver->getMinimumMemoryMapAlignment();
        }
};

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
class SimpleGPUBufferAllocator : public GPUMemoryAllocatorBase
{
        IDriverMemoryBacked::SDriverMemoryRequirements mBufferMemReqs;
        std::pair<uint8_t*,IGPUBuffer*> lastAllocation; //! Make ResizableAddressAllocatorAdaptorBase worry about getting a null DATA buffer when derived does not support!

        inline auto     createAndMapBuffer()
        {
            std::pair<uint8_t*,IGPUBuffer*> retval(nullptr,nullptr);
            retval.second = mDriver->createGPUBufferOnDedMem(mBufferMemReqs,false);

            auto mappingCaps = mBufferMemReqs.mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
            if (mappingCaps)
            {
                auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u,mBufferMemReqs.vulkanReqs.size};
                auto memory = const_cast<IDriverMemoryAllocation*>(retval->getBoundMemory());
                retval.first  = reinterpret_cast<uint8_t*>(memory->mapMemoryRange(static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps),rangeToMap));
            }
            return retval;
        }
    public:
        using GPUMemoryAllocatorBase::GPUMemoryAllocatorBase;

        inline void*    allocate(size_t bytes) noexcept
        {
            assert(!lastAllocation.first && !lastAllocation.second);

            mBufferMemReqs.vulkanReqs.size = bytes;
            lastAllocation = createAndMapBuffer();
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*    reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets) noexcept
        {
            assert(lastAllocation.first==reinterpret_cast<uint8_t*>(addr) && lastAllocation.second);
            // set up new size
            auto oldSize = mBufferMemReqs.vulkanReqs.size;
            mBufferMemReqs.vulkanReqs.size = bytes;
            //allocate new buffer
            auto tmp = createAndMapBuffer();

            //move contents
            auto oldOffset = allocToQueryOffsets.get_align_offset();
            auto newOffset = AddressAllocator::aligned_start_offset(reinterpret_cast<size_t>(tmp.first),allocToQueryOffsets.max_alignment());
            auto copyRangeLen = std::min(oldSize-oldOffset,bytes-newOffset);

            if (lastAllocation.first && tmp.first && // non-null buffers
                (lastAllocation.second->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_READ) &&
                (tmp.second->getBoundMemory()&IDriverMemoryAllocation::EMCAF_WRITE))
            {
                memcpy(tmp.first+newOffset,lastAllocation.first+oldOffset,copyRangeLen);
            }
            else
                mDriver->copyBuffer(lastAllocation.second,tmp.second,oldOffset,newOffset,copyRangeLen);

            //swap the internals of buffers
            lastAllocation.second->pseudoMoveAssign(tmp.second);
            tmp.second->drop();

            //book-keeping and return
            lastAllocation.first = tmp.first;
            return lastAllocation.first;
        }

        inline void     deallocate(void* addr) noexcept
        {
            assert(lastAllocation.first==reinterpret_cast<uint8_t*>(addr) && lastAllocation.second);

            lastAllocation.second->drop();
            lastAllocation = std::pair<uint8_t*,IGPUBuffer*>(nullptr,nullptr);
        }
};

template<class AddressAllocator, bool onlySwapRangesMarkedDirty = false, class CPUAllocator=core::allocator<uint8_t> >
class CResizableDoubleBufferingAllocator : public CDoubleBufferingAllocator<AddressAllocator,onlySwapRangesMarkedDirty,CPUAllocator>
{
    public:
        typedef core::address_allocator_traits<AddressAllocator>                                            alloc_traits;
        typedef typename AddressAllocator::size_type                                                        size_type;

        //! The IDriverMemoryBacked::SDriverMemoryRequirements::size must be identical for both reqs
        template<typename... Args>
        CResizableDoubleBufferingAllocator( IVideoDriver* driver, const IDriverMemoryBacked::SDriverMemoryRequirements& stagingBufferReqs,
                                            const IDriverMemoryBacked::SDriverMemoryRequirements& frontBufferReqs, Args&&... args) : // delegate
                                            CResizableDoubleBufferingAllocator(driver,
                                                                               IDriverMemoryAllocation::MemoryRange(0,stagingBufferReqs.vulkanReqs.size),
                                                                               createAndMapBuffer(driver,stagingBufferReqs), 0u,
                                                                               driver->createGPUBufferOnDedMem(frontBufferReqs,false), std::forward<Args>(args)...)
        {
            Base::mFrontBuffer->drop();
        }

        //! DO NOT USE THIS CONTRUCTOR UNLESS YOU MEAN FOR THE IGPUBufferS TO BE RESIZED AT WILL !
        template<typename... Args>
        CResizableDoubleBufferingAllocator( IVideoDriver* driver, const IDriverMemoryAllocation::MemoryRange& rangeToUse, IGPUBuffer* stagingBuff,
                                            size_t destBuffOffset, IGPUBuffer* destBuff, Args&&... args) :
                                                Base(driver,rangeToUse,stagingBuff,destBuffOffset,destBuff,std::forward<Args>(args)...),
                                                            growPolicy(defaultGrowPolicy), shrinkPolicy(defaultShrinkPolicy)
        {
        }
};

}
}

#endif // __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__


