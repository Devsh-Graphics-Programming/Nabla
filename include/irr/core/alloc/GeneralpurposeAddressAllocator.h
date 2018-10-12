// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_GENERALPURPOSE_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_GENERALPURPOSE_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/math/irrMath.h"

#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{

namespace impl
{

template<typename _size_type>
class GeneralpurposeAddressAllocatorBase
{
    protected:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        GeneralpurposeAddressAllocatorBase(const size_type minBlockSz) : minBlockSize(minBlockSz), minBlockSize_minus1(minBlockSize-1u) {}


        size_type minBlockSize;
        size_type minBlockSize_minus1;

        struct Block
        {
            size_type startOffset;
            size_type endOffset;

            inline size_type    getLength()                     const {return endOffset-startOffset;}
            inline bool         operator<(const Block& other)   const {return startOffset<other.startOffset;}

            inline void         validate(size_type level)
            {
                #ifdef _DEBUG
                assert(getLength()>>level); // in the right free list
                #endif // _DEBUG
            }
        };


        //! Produced blocks can only be larger than `minBlockSize`, it is easier to reason about the correctness and memory boundedness of the allocation algorithm
        inline size_type calcSubAllocation(Block& retval, const Block& block, const size_type bytes, const size_type alignment) const
        {
        #ifdef _DEBUG
            assert(bytes>=minBlockSize && isPoT(alignment));
        #endif // _DEBUG

            retval.startOffset = irr::core::alignUp(block.startOffset,alignment);
            if (block.startOffset!=retval.startOffset)
            {
                auto initialPreceedingBlockSize = retval.startOffset-block.startOffset;
                if (initialPreceedingBlockSize<minBlockSize)
                    retval.startOffset += (minBlockSize_minus1-initialPreceedingBlockSize+alignment)&alignment;
            }

            if (retval.startOffset>block.endOffset)
                return invalid_address;

            retval.endOffset = retval.startOffset+bytes;
            if (retval.endOffset>block.endOffset)
                return invalid_address;

            size_type wastedSpace = block.endOffset-retval.endOffset;
            if (wastedSpace!=size_type(0u) && wastedSpace<minBlockSize)
                return invalid_address;

            return wastedSpace;
        }
};


template<typename _size_type, constexpr bool useBestFitStrategy>
class GeneralpurposeAddressAllocatorStrategy;

template<typename _size_type>
class GeneralpurposeAddressAllocatorStrategy<true> : protected GeneralpurposeAddressAllocatorBase<_size_type>
{
        typedef GeneralpurposeAddressAllocatorBase<_size_type>      Base;
    protected:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);
        typedef typename GeneralpurposeAddressAllocatorBase::Block  Block;

        GeneralpurposeAddressAllocatorStrategy(const size_type minBlockSz) : GeneralpurposeAddressAllocatorBase(minBlockSz) {}


        inline auto blockSearchTargetSize(const size_type bytes, const size_type alignment) const
        {
            return bytes;
        }

        inline auto findSuitableBlock(const size_type bytes, const size_type alignment, const Block* begin, const Block* end) const
        {
            size_type bestWastedSpace = ~size_type(0u);
            std::pair<const Block*,size_type> bestBlock(nullptr,invalid_address);

            for (const Block* it=begin; it!=end; it++)
            {
                Block tmp;
                size_type wastedSpace = calcSubAllocation(tmp,*it,bytes,alignment);
                if (wastedSpace==invalid_address)
                    continue;

                wastedSpace += tmp.startOffset-it->startOffset;
                if (wastedSpace<bestWastedSpace)
                {
                    bestWastedSpace = wastedSpace;
                    bestBlock = std::pair(it,tmp.startOffset);
                }
            }

            return bestBlock;
        }
};

template<typename _size_type>
class GeneralpurposeAddressAllocatorStrategy<false> : protected GeneralpurposeAddressAllocatorBase<_size_type>
{
        typedef GeneralpurposeAddressAllocatorBase<_size_type>      Base;
    protected:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);
        typedef typename GeneralpurposeAddressAllocatorBase::Block  Block;

        GeneralpurposeAddressAllocatorStrategy(const size_type minBlockSz) : GeneralpurposeAddressAllocatorBase(minBlockSz) {}


        inline auto blockSearchTargetSize(const size_type bytes, const size_type alignment) const
        {
            return bytes+std::max(alignment,Base::minBlockSize)+Base::minBlockSize;
        }

        inline auto findSuitableBlock(const size_type bytes, const size_type alignment, const Block* begin, const Block* end) const
        {
            for (const Block* it=begin; it!=end; it++)
            {
                Block tmp;
                size_type wastedSpace = calcSubAllocation(tmp,*it,bytes,alignment);
                if (wastedSpace==invalid_address)
                    continue;

                return std::pair(it,tmp.startOffset);
            }

            return std::pair<const Block*,size_type>(nullptr,invalid_address);
        }
};

}

//! General-purpose allocator, really its like a buddy allocator that supports more sophisticated coalescing
template<typename _size_type, constexpr bool useBestFitStrategy = false>
class GeneralpurposeAddressAllocator : public AddressAllocatorBase<GeneralpurposeAddressAllocator<_size_type>,_size_type>, protected impl::GeneralpurposeAddressAllocatorStrategy<_size_type,useBestFitStrategy>
{
    private:
        typedef impl::GeneralpurposeAddressAllocatorStrategy<_size_type,useBestFitStrategy> AllocStrategy;
        typedef AddressAllocatorBase<GeneralpurposeAddressAllocator<_size_type>,_size_type> Base;

        inline void                     setupFreeListsPointers()
        {
            AllocStrategy::Block* tmp = getCoalesceList()+(bufferSize/AllocStrategy::minBlockSize);
            for (uint32_t j=usingFirstBuffer; j<2u; j++)
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeListStack[i] = tmp;
                tmp += (bufferSize/(AllocStrategy::minBlockSize<<size_type(i))+size_type(1u))/size_type(2u);
            }
        }
    protected:
        //! Lists contain blocks of size < (minBlock<<listIndex)*2 && size >= (minBlock<<listIndex)
        static inline uint32_t          findFreeListInsertIndex(size_type byteSize)
        {
            return findMSB(byteSize/AllocStrategy::minBlockSize);
        }
        static inline uint32_t          findFreeListCount(size_type byteSize)
        {
            return findFreeListInsertIndex(byteSize)+1u;
        }
        inline uint32_t                 findFreeListSearchIndex(size_type byteSize)
        {
            uint32_t retval = findFreeListInsertIndex(byteSize);
            if (retval+1u<freeListCount)
                return retval+1u;
            return retval;
        }
        inline AllocStrategy::Block*    getCoalesceList()
        {
            return reinterpret_cast<AllocStrategy::Block*>(Base::reservedSpace);
        }
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        #define DUMMY_DEFAULT_CONSTRUCTOR GeneralpurposeAddressAllocator() : AllocStrategy(0xdeadbeefu), bufferSize(0u),freeSize(0u), allocatedSize(0u), blocksToCoalesceCount(0u), freeListCount(0u), usingFirstBuffer(1u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~GeneralpurposeAddressAllocator() {}

        GeneralpurposeAddressAllocator(void* reservedSpc, void* buffer, size_type maxAllocatableAlignment, size_type bufSz, size_type minBlockSz) noexcept :
                    Base(reservedSpc,buffer,maxAllocatableAlignment), AllocStrategy(minBlockSz), bufferSize(bufSz-Base::alignOffset), freeSize(bufferSize),
                    allocatedSize(0u), blocksToCoalesceCount(0u), freeListCount(findFreeListCount(bufSz-Base::alignOffset)), usingFirstBuffer(1u)
        {
            assert(bufferSize>=Base::alignOffset+AllocStrategy::minBlockSize && bufferSize<invalid_address && static_cast<size_t>(bufferSize) < (size_t(1u)<<maxListLevels));

            setupFreeListsPointers();
            reset();
        }

        GeneralpurposeAddressAllocator(const GeneralpurposeAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) noexcept :
                    Base(other,newReservedSpc,newBuffer,newBuffSz), AllocStrategy(other.blockSize)
        {
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            for (size_type i=0u; i<freeStackCtr; i++)
                freeStack[i] = (blockCount-1u-i)*blockSize+Base::alignOffset;

            for (size_type i=0; i<other.freeStackCtr; i++)
            {
                size_type freeEntry = other.addressToBlockID(reinterpret_cast<size_type*>(other.reservedSpace)[i]);

                if (freeEntry<blockCount)
                    freeStack[freeStackCtr++] = freeEntry*blockSize+Base::alignOffset;
            }

            if (other.bufferStart&&Base::bufferStart)
            {
                memmove(reinterpret_cast<uint8_t*>(Base::bufferStart)+Base::alignOffset,
                        reinterpret_cast<uint8_t*>(other.bufferStart)+other.alignOffset,
                        std::min(blockCount,other.blockCount)*blockSize);
            }
        }

        GeneralpurposeAddressAllocator& operator=(GeneralpurposeAddressAllocator&& other)
        {
            Base::operator=(std::move(other));
            AllocStrategy::operator=(std::move(other));
            //blockSize = other.blockSize;
            return *this;
        }


        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (alignment>Base::maxRequestableAlignment || bytes==0u)
                return invalid_address;

            bytes = std::max(bytes,AllocStrategy::minBlockSize);
            if (bytes>freeSize)
                return invalid_address;

            std::pair<size_type,AllocStrategy::Block> found;
            for (auto i=0u; i<2u; i++)
            {
                // get finding index
                uint32_t level = findFreeListSearchIndex(AllocStrategy::blockSearchTargetSize(bytes,alignment));
                // try to find block going up the hierarchy
                for (; level<freeListCount; level++)
                {
                    auto stackBegin = freeListStack[level];
                    auto found = AllocStrategy::findSuitableBlock(bytes,alignment,stackBegin,stackBegin+freeListStackCtr[level]);
                }

                // find block
                found = AllocStrategy::findSuitableBlock(bytes,alignment,freeListStack,freeListStackCtr);

                // if not found first time, then defragment, else break
                if (found.first!=invalid_address || i)
                    break;

                defragment();
            }

            // not found anything
            if (found.first==invalid_address)
                return invalid_address;

            // splice block and insert parts onto free list
            found.second.validate();
            if (found.first+bytes!=found.second.endOffset)
            {
                AllocStrategy::Block tmp{found.first+bytes,found.second.endOffset};
                tmp.validate();
                auto level = findFreeListInsertIndex(tmp.endOffset-tmp.startOffset);
                freeListStack[level][freeListStackCtr[level]++] = tmp;
            }
            if (found.first!=found.second.startOffset)
            {
                AllocStrategy::Block tmp{found.second.startOffset,found.first};
                tmp.validate();
                auto level = findFreeListInsertIndex(tmp.endOffset-tmp.startOffset);
                freeListStack[level][freeListStackCtr[level]++] = tmp;
            }

            freeSize        -= bytes;
            allocatedSize   += bytes;

            return found.first+Base::alignOffset;
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
            if (bytes<AllocStrategy::minBlockSize)
                bytes = AllocStrategy::minBlockSize;
#ifdef _DEBUG
            assert(addr>=Base::alignOffset && addr+bytes+Base::alignOffset<bufferSize);
#endif // _DEBUG

            getCoalesceList()[blocksToCoalesceCount++] = AllocStrategy::Block{addr,addr+bytes};

            allocatedSize -= bytes;
        }

        inline void             reset()
        {
            freeSize                = bufferSize;
            allocatedSize           = 0u;
            blocksToCoalesceCount   = 0u;

            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
                freeListStackCtr[i] = 0u;

            assert(findFreeListInsertIndex(bufferSize)+1u==freeListCount);
            freeListStack[freeListCount-1u][freeListStackCtr[freeListCount-1u]++] = AllocStrategy::Block{0u,bufferSize};
        }

        //! conservative estimate, does not account for space lost to alignment or actual block size
        inline size_type        max_size() const noexcept
        {
            for (decltype(freeListCount) i=freeListCount; i; i--)
            {
                size_type level = i-1u;
                if (freeListStackCtr[level])
                    return size_type(AllocStrategy::minBlockSize)<<level;
            }

            return 0u;
        }

        inline size_type        safe_shrink_size(size_type byteBound=0u, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            size_type retval = get_total_size()-Base::alignOffset;
            if (byteBound>=retval)
                return Base::safe_shrink_size(byteBound,newBuffAlignmentWeCanGuarantee);

            if (get_free_size()==0u)
                return Base::safe_shrink_size(retval,newBuffAlignmentWeCanGuarantee);

            //now increase byteBound by taking into account fragmentation
            retval = defragment();

            return Base::safe_shrink_size(std::max(retval,byteBound),newBuffAlignmentWeCanGuarantee);
        }


        static inline size_type reserved_size(size_type bufSz, size_type maxAlignment, size_type blockSz) noexcept
        {
            size_type reserved = bufSz/blockSz;
            for (size_type i=0u; i<findFreeListCount(blockSz); i++)
                reserved += alignUp(bufSz/(blockSz<<i)+size_type(1u),size_type(2u));
            return reserved;
        }
        static inline size_type reserved_size(size_type bufSz, const GeneralpurposeAddressAllocator<_size_type>& other) noexcept
        {
            return reserved_size(bufSz,other.maxRequestableAlignment,other.blockSize);
        }

        inline size_type        get_free_size() const noexcept
        {
            return freeSize; //allocatable decrement when taking block off free list, increment when putting on
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return allocatedSize; // increment when allocating, decrement when deallocating
        }
        inline size_type        get_total_size() const noexcept
        {
            return bufferSize+Base::alignOffset;
        }
    protected:
        size_type               bufferSize;
        size_type               freeSize;
        size_type               allocatedSize;
        size_type               blocksToCoalesceCount;
        uint32_t                freeListCount;
        uint32_t                usingFirstBuffer;

        constexpr static size_t maxListLevels = std::min(sizeof(size_type)*sizeof(uint8_t),size_t(59u));
        size_type               freeListStackCtr[maxListLevels];
        AllocStrategy::Block*   freeListStack[maxListLevels];

        inline void             insertFreeBlock()
        {

        }
        inline size_type        defragment()
        {
            auto coalesceList = getCoalesceList();
            const AllocStrategy::Block* coalesceListEnd = coalesceList+blocksToCoalesceCount;
            std::sort(coalesceList,const_cast<AllocStrategy::Block*>(coalesceListEnd));

            AllocStrategy::Block* freeList[maxListLevels];
            const AllocStrategy::Block* freeListEnd[maxListLevels];
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeList[i] = freeListStack[i];
                freeListEnd[i] = freeList+freeListStackCtr[i];
                std::sort(freeList[i],const_cast<AllocStrategy::Block*>(freeListEnd[i]));

                freeListStackCtr[i] = 0u;
            }

            AllocStrategy::Block* sortedScratchListPtr = coalesceList+(bufferSize/AllocStrategy::minBlockSize);
            const AllocStrategy::Block* sortedScratchListBegin = sortedScratchListPtr;

            AllocStrategy::Block lastBlock{0u,0u};
            auto minimum = findMinimum(freeList,freeListEnd,coalesceList,coalesceListEnd);
            while (minimum!=invalid_address)
            {
                // find next free block and pop it
                AllocStrategy::Block nextBlock;
                if (minimum<freeListCount)
                    nextBlock = *(freeList[minimum]++);
                else
                    nextBlock = *(coalesceList++);

                // check if broke continuity
                if (nextBlock.startOffset!=lastBlock.endOffset)
                {
                    auto blockLen = lastBlock.endOffset-lastBlock.startOffset;
                    // put old on correct free list
                    if (blockLen)
                        freeListStack[freeListStackCtr[findFreeListInsertIndex(blockLen)]++] = lastBlock;

                    lastBlock.startOffset = nextBlock.startOffset;
                }

                lastBlock.endOffset = nextBlock.endOffset;
                minimum = findMinimum(freeList,freeListEnd,coalesceList,coalesceListEnd);
            }
            auto blockLen = lastBlock.endOffset-lastBlock.startOffset;
            // put old on correct free list
            if (blockLen)
            {
                freeListStack[freeListStackCtr[findFreeListInsertIndex(blockLen)]++] = lastBlock;
                if (lastBlock.endOffset==bufferSize)
                    return lastBlock.startOffset;
            }

            return bufferSize;
        }
    private:
        //! Return index of freelist or one past the end for the coalesce list, or invalid_address for nothing
        inline decltype(freeListCount)  findMinimum(AllocStrategy::Block** listOfLists, const AllocStrategy::Block** listOfListsEnd,
                                                    AllocStrategy::Block* otherList, AllocStrategy::Block* otherListEnd)
        {
            size_type               minval = invalid_address;
            decltype(freeListCount) retval = invalid_address;

            for (decltype(freeListCount) i=0; i<freeListCount; i++)
            {
                if (listOfLists[i]==listOfListsEnd[i] || listOfLists[i]->startOffset>=minval)
                    continue;

                minval = listOfLists[i]->startOffset;
                retval = i;
            }

            if (otherList==otherListEnd || otherList->startOffset>=minval)
                return retval;

            return freeListCount;
        }
};


}
}

#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

namespace irr
{
namespace core
{

// aliases
template<typename size_type>
using GeneralpurposeAddressAllocatorST = GeneralpurposeAddressAllocator<size_type>;

template<typename size_type, class BasicLockable>
using GeneralpurposeAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<GeneralpurposeAddressAllocator<size_type>,BasicLockable>;

}
}

#endif // __IRR_GENERALPURPOSE_ADDRESS_ALLOCATOR_H_INCLUDED__


