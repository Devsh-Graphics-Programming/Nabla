#ifndef __IRR_STREAMING_GPUBUFFER_ALLOCATOR_H__
#define __IRR_STREAMING_GPUBUFFER_ALLOCATOR_H__

#include "irr/video/SimpleGPUBufferAllocator.h"

namespace irr
{
namespace video
{

// a really crappy allocator that only supports one allocation at a time
class StreamingGPUBufferAllocator : protected SimpleGPUBufferAllocator
{
    protected:
        std::pair<uint8_t*,IGPUBuffer*>                 lastAllocation;

        inline uint8_t* mapWholeBuffer(IGPUBuffer* buff) noexcept
        {
            auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u,buff->getSize()};
            auto memory = const_cast<IDriverMemoryAllocation*>(buff->getBoundMemory());
            auto mappingCaps = memory->getMappingCaps()&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
            return reinterpret_cast<uint8_t*>(memory->mapMemoryRange(static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps),rangeToMap));
        }
    public:
        StreamingGPUBufferAllocator(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) :
                        SimpleGPUBufferAllocator(inDriver,bufferReqs), lastAllocation(nullptr,nullptr)
        {
            assert(mBufferMemReqs.mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE); // have to have mapping access to the buffer!
        }

        inline void*        allocate(size_t bytes) noexcept
        {
        #ifdef _DEBUG
            assert(!lastAllocation.first && !lastAllocation.second);
        #endif // _DEBUG
            lastAllocation.second = SimpleGPUBufferAllocator::allocate(bytes);
            lastAllocation.first = mapWholeBuffer(lastAllocation.second);
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets, bool copyBuffers=true) noexcept
        {
        #ifdef _DEBUG
            assert(lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG

            // set up new size and allocate new buffer
            auto oldSize = lastAllocation.second->getSize();
            IGPUBuffer* newBuff = SimpleGPUBufferAllocator::allocate(bytes);
            uint8_t* newPointer = mapWholeBuffer(newBuff);

            auto newOffset = AddressAllocator::aligned_start_offset(reinterpret_cast<size_t>(newPointer),allocToQueryOffsets.max_alignment());

            //move contents
            if (copyBuffers)
            {
                // only first buffer is bound to allocator
                auto oldOffset = allocToQueryOffsets.get_align_offset();
                auto copyRangeLen = std::min(oldSize-oldOffset,bytes-newOffset);

                if (addr && (lastAllocation.second->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_READ) &&
                    (newBuff->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_WRITE)) // can read from old and write to new
                {
                    memcpy(newPointer+newOffset,lastAllocation.first+oldOffset,copyRangeLen);
                }
                else
                    copyBufferWrapper(lastAllocation.second,newBuff,oldOffset,newOffset,copyRangeLen);
            }

            //swap the internals of buffers
            const_cast<IDriverMemoryAllocation*>(lastAllocation.second->getBoundMemory())->unmapMemory();
            lastAllocation.second->pseudoMoveAssign(newBuff);
            newBuff->drop();

            //book-keeping and return
            lastAllocation.first = newPointer;
            return newPointer;
        }

        inline void         deallocate(void* addr) noexcept
        {
            #ifdef _DEBUG
            assert(lastAllocation.first==addr && lastAllocation.second);
            #endif // _DEBUG
            lastAllocation.first = nullptr;
            const_cast<IDriverMemoryAllocation*>(lastAllocation.second->getBoundMemory())->unmapMemory();
            SimpleGPUBufferAllocator::deallocate(lastAllocation.second);
            lastAllocation.second = nullptr;
        }


        // extras
        inline IGPUBuffer*  getAllocatedBuffer()
        {
            return lastAllocation.second;
        }

        inline void*        getAllocatedPointer()
        {
            return lastAllocation.first;
        }

        //to expose base functions again
        IVideoDriver*   getDriver() noexcept {return SimpleGPUBufferAllocator::getDriver();}
};

}
}

#endif // __IRR_STREAMING_GPUBUFFER_ALLOCATOR_H__
