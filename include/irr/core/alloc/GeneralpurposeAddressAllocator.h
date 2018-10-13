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

            inline void         validate(size_type level)       const
            {
                #ifdef _DEBUG
                assert(getLength()>>level); // in the right free list
                #endif // _DEBUG
            }
        };


        //! Produced blocks can only be larger than `minBlockSize`, it is easier to reason about the correctness and memory boundedness of the allocation algorithm
        inline size_type calcSubAllocation(Block& retval, const Block* block, const size_type bytes, const size_type alignment) const
        {
        #ifdef _DEBUG
            assert(bytes>=minBlockSize && isPoT(alignment));
        #endif // _DEBUG

            retval.startOffset = irr::core::alignUp(block->startOffset,alignment);
            if (block->startOffset!=retval.startOffset)
            {
                auto initialPreceedingBlockSize = retval.startOffset-block->startOffset;
                if (initialPreceedingBlockSize<minBlockSize)
                    retval.startOffset += (minBlockSize_minus1-initialPreceedingBlockSize+alignment)&alignment;
            }

            if (retval.startOffset>block->endOffset)
                return invalid_address;

            retval.endOffset = retval.startOffset+bytes;
            if (retval.endOffset>block->endOffset)
                return invalid_address;

            size_type wastedSpace = block->endOffset-retval.endOffset;
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


        inline auto findAndPopSuitableBlock(const size_type bytes, const size_type alignment, AllocStrategy::Block** listOfLists, size_type* listCounters)
        {
            size_type bestWastedSpace = ~size_type(0u);
            std::tuple<Block,Block*,decltype(freeListCount)> bestBlock(AllocStrategy::Block{invalid_address,invalid_address},nullptr,freeListCount);

            // using findFreeListInsertIndex on purpose
            for (uint32_t level=findFreeListInsertIndex(bytes); level<freeListCount && bestWastedSpace; level++)
            for (Block* it=listOfLists[level]; it!=(listOfLists[level]+listCounters[level]) && bestWastedSpace; it++)
            {
                Block tmp;
                size_type wastedSpace = calcSubAllocation(tmp,it,bytes,alignment);
                if (wastedSpace==invalid_address)
                    continue;

                wastedSpace += tmp.startOffset-it->startOffset;
                if (wastedSpace>=bestWastedSpace)
                    continue;

                bestWastedSpace = wastedSpace;
                bestBlock = std::tuple(tmp,it,level);
            }

            AllocStrategy::Block sourceBlock{invalid_address,invalid_address}
            // if found something
            Block* out = bestBlock.get<1u>();
            if (out)
            {
                sourceBlock = *out;
                for (Block* in=out+1u; in!=(listOfLists[level]+listCounters[level]); in++,out++)
                    *out = *in;
                listCounters[level]--;
            }

            return std::pair(bestBlock.get<0u>(),sourceBlock);
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


        inline auto findAndPopSuitableBlock(const size_type bytes, const size_type alignment, AllocStrategy::Block** listOfLists, size_type* listCounters)
        {
            for (uint32_t level=findFreeListSearchIndex(bytes+std::max(alignment,Base::minBlockSize)+Base::minBlockSize); level<freeListCount; level++)
            {
                // have any free blocks
                if (!listCounters[level])
                    continue;

                // pop off the top
                const Block& popped = listOfLists[level][--listCounters[level]];
                Block tmp;
                size_type wastedSpace = calcSubAllocation(tmp,popped,bytes,alignment);
                // if had a block large enough for us with padding then must be able to allocate
                assert(wastedSpace!=invalid_address);

                return std::pair(tmp,popped);
            }

            return std::pair(AllocStrategy::Block{invalid_address,invalid_address},AllocStrategy::Block{invalid_address,invalid_address});
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

        inline void                     setupFreeLists()
        {
            AllocStrategy::Block* tmp = reinterpret_cast<AllocStrategy::Block*>(Base::reservedSpace);
            for (uint32_t j=usingFirstBuffer; j<2u; j++)
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeListStack[i] = tmp;
                tmp += bufferSize/(AllocStrategy::minBlockSize<<size_type(i));
                if (i)
                    continue;
                tmp++; // base level dwarf-blocks
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
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        //! TODO: Constructors and Asserts
        #define DUMMY_DEFAULT_CONSTRUCTOR GeneralpurposeAddressAllocator() : AllocStrategy(0xdeadbeefu), bufferSize(0u), freeSize(0u), freeListCount(0u), usingFirstBuffer(1u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~GeneralpurposeAddressAllocator() {}

        GeneralpurposeAddressAllocator(void* reservedSpc, void* buffer, size_type maxAllocatableAlignment, size_type bufSz, size_type minBlockSz) noexcept :
                    Base(reservedSpc,buffer,maxAllocatableAlignment), AllocStrategy(minBlockSz), bufferSize(bufSz-Base::alignOffset), freeSize(bufferSize),
                    freeListCount(findFreeListCount(bufferSize)), usingFirstBuffer(1u)
        {
            assert(bufferSize>=Base::alignOffset+AllocStrategy::minBlockSize && bufferSize<invalid_address && static_cast<size_t>(bufferSize) < (size_t(1u)<<maxListLevels));

            setupFreeLists();
            reset();
        }

        GeneralpurposeAddressAllocator(const GeneralpurposeAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) noexcept :
                    Base(other,newReservedSpc,newBuffer,newBuffSz), AllocStrategy(other.blockSize),  bufferSize(bufSz-Base::alignOffset),
                    freeSize(bufferSize-other.get_allocated_size()), freeListCount(findFreeListCount(bufferSize)), usingFirstBuffer(1u)
        {
        #ifdef _DEBUG
            assert(bufferSize>=other.get_allocated_size() && (bufferSize<=other.bufferSize||bufferSize>=other.bufferSize+other.minBlockSize));
        #endif // _DEBUG
            setupFreeLists();

            bool needToCreateNewFreeBlock = bufferSize>other.bufferSize;
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            for (size_type j=0u; j<other.freeListStackCtr[i]; j++)
            {
                AllocStrategy::Block tmp = other.freeListStack[i][j];
                if (tmp.startOffset<bufferSize)
                {
                    if (needToCreateNewFreeBlock && tmp.endOffset==other.bufferSize)
                    {
                        tmp.endOffset = bufferSize;
                        needToCreateNewFreeBlock = false;
                    }
                    else if (tmp.endOffset>bufferSize)
                        tmp.endOffset = bufferSize;

                    insertFreeBlock(tmp);
                }
            }

            if (needToCreateNewFreeBlock)
                insertFreeBlock(AllocStrategy::Block{other.bufferSize,bufferSize});

            //! TODO: copy buffer data
        }

        GeneralpurposeAddressAllocator& operator=(GeneralpurposeAddressAllocator&& other)
        {
            Base::operator=(std::move(other));
            AllocStrategy::operator=(std::move(other));
            bufferSize      = other.bufferSize;
            freeSize        = other.freeSize;
            freeListCount   = other.freeListCount;
            usingFirstBuffer= other.usingFirstBuffer;

            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeListStackCtr[i] = other.freeListStackCtr[i];
                freeListStack[i] = other.freeListStack[i];
            }
            return *this;
        }


        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (alignment>Base::maxRequestableAlignment || bytes==0u)
                return invalid_address;

            bytes = std::max(bytes,AllocStrategy::minBlockSize);
            if (bytes>freeSize)
                return invalid_address;

            std::pair<AllocStrategy::Block,AllocStrategy::Block> found;
            for (auto i=0u; i<2u; i++)
            {
                found = AllocStrategy::findAndPopSuitableBlock(bytes,alignment,freeListStack,freeListStackCtr);

                // if not found first time, then defragment, else break
                if (found.first.startOffset!=invalid_address || i)
                    break;

                defragment();
            }

            // not found anything
            if (found.first.startOffset==invalid_address)
                return invalid_address;

            // splice block and insert parts onto free list
            if (found.first.endOffset!=found.second.endOffset)
                insertFreeBlock(AllocStrategy::Block{found.first.endOffset,found.second.endOffset});
            if (found.first.startOffset!=found.second.startOffset)
                insertFreeBlock(AllocStrategy::Block{found.second.startOffset,found.first.startOffset});

            freeSize -= bytes;

            return found.first.startOffset+Base::alignOffset;
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
            bytes = std::max(bytes,AllocStrategy::minBlockSize);
#ifdef _DEBUG
            assert(addr>=Base::alignOffset && addr+bytes<=bufferSize+Base::alignOffset && bytes+freeSize<=bufferSize);
#endif // _DEBUG

            addr -= Base::alignOffset;
            insertFreeBlock(AllocStrategy::Block{addr,addr+bytes});
            freeSize += bytes;
        }

        inline void             reset()
        {
            freeSize = bufferSize;
            insertFreeBlock(AllocStrategy::Block{0u,bufferSize});
        }

        //! Conservative estimate, max_size() gives largest size we are sure to be able to allocate
        inline size_type        max_size() const noexcept
        {
            const auto maxWastedSpace = std::max(Base::maxRequestableAlignment,AllocStrategy::minBlockSize)+AllocStrategy::minBlockSize;
            for (decltype(freeListCount) i=freeListCount; i>findFreeListSearchIndex(maxWastedSpace); i--)
            {
                size_type level = i-1u;
                if (freeListStackCtr[level])
                    return (size_type(AllocStrategy::minBlockSize)<<level)-maxWastedSpace;
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
            for (size_type i=0u; i<findFreeListCount(blockSz); i++)
                reserved += (bufSz/(blockSz<<i))*size_type(2u);
            return reserved+2u; // +2 for lowest level partial blocks at the end
        }
        static inline size_type reserved_size(size_type bufSz, const GeneralpurposeAddressAllocator<_size_type>& other) noexcept
        {
            return reserved_size(bufSz,other.maxRequestableAlignment,other.blockSize);
        }

        inline size_type        get_free_size() const noexcept
        {
            return freeSize; // decrement when allocation, increment when freeing
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return bufferSize-freeSize;
        }
        inline size_type        get_total_size() const noexcept
        {
            return bufferSize+Base::alignOffset;
        }
    protected:
        size_type               bufferSize;
        size_type               freeSize;
        uint32_t                freeListCount;
        uint32_t                usingFirstBuffer;

        constexpr static size_t maxListLevels = std::min(sizeof(size_type)*sizeof(uint8_t),size_t(59u));
        size_type               freeListStackCtr[maxListLevels];
        AllocStrategy::Block*   freeListStack[maxListLevels];

        inline void             insertFreeBlock(const AllocStrategy::Block& block)
        {
            auto level = findFreeListInsertIndex(block.endOffset-block.startOffset);
            block.validate(level);
            freeListStack[level][freeListStackCtr[level]++] = block;
        }
        inline size_type        defragment()
        {
            AllocStrategy::Block* freeListOld[maxListLevels];
            const AllocStrategy::Block* freeListOldEnd[maxListLevels];
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeListOld[i] = freeListStack[i];
                freeListOldEnd[i] = freeListOld[i]+freeListStackCtr[i];
                std::sort(freeListOld[i],const_cast<AllocStrategy::Block*>(freeListOldEnd[i]));
            }

            // swap free-list buffers
            usingFirstBuffer = usingFirstBuffer ? 0u:1u;
            setupFreeLists();

            // begin the coalesce
            AllocStrategy::Block lastBlock{0u,0u};
            auto minimum = findMinimum(freeListOld,freeListOldEnd);
            while (minimum!=freeListCount)
            {
                // find next free block and pop it
                const AllocStrategy::Block* nextBlock = freeListOld[minimum]++;

                // check if broke continuity
                if (nextBlock->startOffset!=lastBlock.endOffset)
                {
                    // put old on correct free list
                    if (lastBlock.getLength())
                        insertFreeBlock(lastBlock);

                    lastBlock.startOffset = nextBlock->startOffset;
                }

                lastBlock.endOffset = nextBlock->endOffset;
                minimum = findMinimum(freeListOld,freeListOldEnd);
            }
            // put last block on correct free list
            if (lastBlock.getLength())
            {
                insertFreeBlock(lastBlock);
                if (lastBlock.endOffset==bufferSize)
                    return lastBlock.startOffset;
            }

            return bufferSize;
        }
    private:
        //! Return index of freelist or one past the end for nothing
        inline decltype(freeListCount)  findMinimum(const AllocStrategy::Block** listOfLists, const AllocStrategy::Block** listOfListsEnd)
        {
            size_type               minval = ~size_type(0u);
            decltype(freeListCount) retval = freeListCount;

            for (decltype(freeListCount) i=0; i<freeListCount; i++)
            {
                if (listOfLists[i]==listOfListsEnd[i] || listOfLists[i]->startOffset>=minval)
                    continue;

                minval = listOfLists[i]->startOffset;
                retval = i;
            }

            return retval;
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


