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
        //types
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);
        struct Block
        {
            size_type startOffset;
            size_type endOffset;

            inline size_type    getLength()                     const {return endOffset-startOffset;}
            inline bool         operator<(const Block& other)   const {return startOffset<other.startOffset;}

            inline void         validate(size_type level)       const
            {
                #ifdef _IRR_DEBUG
                assert(getLength()>>level); // in the right free list
                #endif // _IRR_DEBUG
            }
        };
        static inline uint32_t  findFreeListCount(size_type byteSize, size_type minBlockSz) noexcept
        {
            return findFreeListInsertIndex(byteSize,minBlockSz)+1u;
        }


        // constructors
        GeneralpurposeAddressAllocatorBase(size_type bufSz, size_type minBlockSz) noexcept :
            bufferSize(bufSz), freeSize(0u), freeListCount(findFreeListCount(bufferSize,minBlockSz)),
            usingFirstBuffer(0u), minBlockSize(minBlockSz){}

        GeneralpurposeAddressAllocatorBase(size_type newBuffSz, GeneralpurposeAddressAllocatorBase&& other, void* newReservedSpc) noexcept :
            bufferSize(newBuffSz), freeSize(0u), freeListCount(findFreeListCount(bufferSize,other.minBlockSize)),
            usingFirstBuffer(0u), minBlockSize(other.minBlockSize)
        {
            swapFreeLists(newReservedSpc);
            // first, insert new block or trim existing
            if (newBuffSz<other.bufferSize) // trim
            {
                bool notFoundTheSlab = true;
                for (auto i=freeListCount; notFoundTheSlab&&i<other.freeListCount; i++)
                for (size_type j=0u; j<other.freeListStackCtr[i]; j++)
                {
                    const auto& block = other.freeListStack[i][j];
                    if (block.startOffset>bufferSize)
                        continue;
                    #ifdef _IRR_DEBUG
                    assert(block.endOffset>=bufferSize);
                    #endif // _IRR_DEBUG
                    insertFreeBlock({block.startOffset,bufferSize});
                    #ifndef _IRR_DEBUG
                    notFoundTheSlab = false;
                    #endif // _IRR_DEBUG
                }
            }
            else if (newBuffSz>other.bufferSize) // insert new
                insertFreeBlock({other.bufferSize,bufferSize});
            // then copy the existing free-blocks across
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                if (i<other.freeListCount)
                {
                    for (size_type j=0u; j<other.freeListStackCtr[i]; j++)
                    {
                        const auto& block = other.freeListStack[i][j];
                        freeListStack[i][freeListStackCtr[i]++]= block;
                        freeSize += block.getLength();
                    }
                }
                other.freeListStackCtr[i] = invalid_address;
                other.freeListStack[i] = nullptr;
            }
            other.bufferSize = invalid_address;
            other.freeSize = invalid_address;
            other.freeListCount = invalid_address;
            other.usingFirstBuffer = invalid_address;
            other.minBlockSize = invalid_address;
        }

        virtual ~GeneralpurposeAddressAllocatorBase() {}


        GeneralpurposeAddressAllocatorBase& operator=(GeneralpurposeAddressAllocatorBase&& other)
        {
            std::swap(bufferSize,other.bufferSize);
            std::swap(freeSize,other.freeSize);
            std::swap(freeListCount,other.freeListCount);
            std::swap(usingFirstBuffer,other.usingFirstBuffer);

            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeListStackCtr[i] = invalid_address;
                freeListStack[i] = nullptr;
                std::swap(freeListStackCtr[i],other.freeListStackCtr[i]);
                std::swap(freeListStack[i],other.freeListStack[i]);
            }
            return *this;
        }


        // members
        size_type               bufferSize;
        size_type               freeSize;
        uint32_t                freeListCount;
        uint32_t                usingFirstBuffer;
        size_type               minBlockSize;

        constexpr static size_t maxListLevels = (sizeof(size_type)*8u)<size_t(59ull) ? (sizeof(size_type)*8u):size_t(59ull);
        size_type               freeListStackCtr[maxListLevels];
        Block*                  freeListStack[maxListLevels];


        //methods
        inline bool                 is_double_free(size_type addr, size_type bytes) const noexcept
        {
            size_type totalFree = 0u;
            for (uint32_t level=0u; level<freeListCount; level++)
            for (uint32_t i=0u; i<freeListStackCtr[level]; i++)
            {
                const Block& freeb = freeListStack[level][i];
                totalFree += freeb.getLength();
                if (addr>=freeb.endOffset)
                    continue;

                if (addr+bytes<=freeb.startOffset)
                    continue;

                return true;
            }
            #ifdef _IRR_DEBUG
            assert(freeSize==totalFree);
            #endif // _IRR_DEBUG
            return false;
        }
        inline uint32_t          findFreeListInsertIndex(size_type byteSize) const noexcept
        {
            return findFreeListInsertIndex(byteSize,minBlockSize);
        }
        inline uint32_t          findFreeListSearchIndex(size_type byteSize) const noexcept
        {
            uint32_t retval = findFreeListInsertIndex(byteSize);
            if (retval+1u<freeListCount)
                return retval+1u;
            return retval;
        }

        inline void              swapFreeLists(void* startPtr) noexcept
        {
            freeSize = 0u;
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
                freeListStackCtr[i] = 0u;

            usingFirstBuffer = usingFirstBuffer ? 0u:1u;

            Block* tmp = reinterpret_cast<Block*>(startPtr);
            for (uint32_t j=usingFirstBuffer; j<2u; j++)
            for (decltype(freeListCount) i=0u; i<freeListCount; i++)
            {
                freeListStack[i] = tmp;
                tmp += bufferSize/(minBlockSize<<size_type(i));
                if (i)
                    continue;
                tmp++; // base level dwarf-blocks
            }
        }

        inline void             insertFreeBlock(const Block& block)
        {
            auto len = block.getLength();
        #ifdef _IRR_DEBUG
            if (len<minBlockSize)
                assert(false);
        #endif // _IRR_DEBUG
            auto level = findFreeListInsertIndex(len);
            block.validate(level);
            freeListStack[level][freeListStackCtr[level]++] = block;
        #ifdef _IRR_DEBUG
            assert(freeListStackCtr[level]<=bufferSize/(minBlockSize<<level)+(level==0u ? 1u:0u));
        #endif // _IRR_DEBUG
            freeSize += len;
        }

        //! Produced blocks can only be larger than `minBlockSize`, so it's easier to reason about the correctness and memory boundedness of the allocation algorithm
        inline size_type calcSubAllocation(Block& retval, const Block* block, const size_type bytes, const size_type alignment) const
        {
        #ifdef _IRR_DEBUG
            assert(bytes>=minBlockSize);
        #endif // _IRR_DEBUG

            retval.startOffset = core::roundUp(block->startOffset,alignment);

            if (block->startOffset!=retval.startOffset)
            {
                auto initialPreceedingBlockSize = retval.startOffset-block->startOffset;
                if (initialPreceedingBlockSize<minBlockSize)
                    retval.startOffset += core::roundUp(minBlockSize-initialPreceedingBlockSize,alignment);
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


        //! Return index of freelist or one past the end for nothing
        inline decltype(freeListCount)  findMinimum(const Block* const* listOfLists, const Block* const* listOfListsEnd) noexcept
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


        inline size_type        getBiggestPossibleFrontWaste(size_type alignment) const noexcept
        {
            return alignment+minBlockSize-1u;
        }

    private:
        //! Lists contain blocks of size < (minBlock<<listIndex)*2 && size >= (minBlock<<listIndex)
        static inline uint32_t  findFreeListInsertIndex(size_type byteSize, size_type minBlockSz) noexcept
        {
#ifdef _IRR_DEBUG
            assert(byteSize>=minBlockSz); // logic fail
#endif // _IRR_DEBUG
            return findMSB(byteSize/minBlockSz);
        }
};


template<typename _size_type, bool useBestFitStrategy>
class GeneralpurposeAddressAllocatorStrategy;

template<typename _size_type>
class GeneralpurposeAddressAllocatorStrategy<_size_type,true> : protected GeneralpurposeAddressAllocatorBase<_size_type>
{
        typedef GeneralpurposeAddressAllocatorBase<_size_type>  Base;
    protected:
        typedef typename Base::Block                            Block;
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        using Base::Base;


        inline std::pair<Block,Block> findAndPopSuitableBlock(const size_type bytes, const size_type alignment) noexcept
        {
            size_type bestWastedSpace = ~size_type(0u);
            std::tuple<Block,Block*,decltype(Base::freeListCount)> bestBlock{Block{invalid_address,invalid_address},nullptr,Base::freeListCount};

            // using findFreeListInsertIndex on purpose
            for (uint32_t level=Base::findFreeListInsertIndex(bytes); level<Base::freeListCount && bestWastedSpace; level++)
            for (Block* it=Base::freeListStack[level]; it!=(Base::freeListStack[level]+Base::freeListStackCtr[level]) && bestWastedSpace; it++) // TODO: Iterate in reverse
            {
                Block tmp;
                size_type wastedSpace = Base::calcSubAllocation(tmp,it,bytes,alignment);
                if (wastedSpace==invalid_address)
                    continue;

                wastedSpace += tmp.startOffset-it->startOffset;
                if (wastedSpace>=bestWastedSpace)
                    continue;

                bestWastedSpace = wastedSpace;
                bestBlock = std::tuple<Block,Block*,decltype(Base::freeListCount)>{tmp,it,level};
            }

            Block sourceBlock{invalid_address,invalid_address};
            // if found something
            Block* out = std::get<1u>(bestBlock);
            if (out)
            {
                sourceBlock = *out;
                auto level = std::get<2u>(bestBlock);
                for (Block* in=out+1u; in!=(Base::freeListStack[level]+Base::freeListStackCtr[level]); in++,out++)
                    *out = *in;
                Base::freeListStackCtr[level]--;
                Base::freeSize -= sourceBlock.getLength();
            }

            return std::pair<Block,Block>(std::get<0u>(bestBlock),sourceBlock);
        }
};

template<typename _size_type>
class GeneralpurposeAddressAllocatorStrategy<_size_type,false> : protected GeneralpurposeAddressAllocatorBase<_size_type>
{
        typedef GeneralpurposeAddressAllocatorBase<_size_type>  Base;
    protected:
        typedef typename Base::Block                            Block;
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        using Base::Base;


        inline std::pair<Block,Block>   findAndPopSuitableBlock(const size_type bytes, const size_type alignment) noexcept
        {
            auto maxWastedSpace = Base::getBiggestPossibleFrontWaste(alignment)+Base::minBlockSize;
            for (uint32_t level=Base::findFreeListSearchIndex(bytes+maxWastedSpace); level<Base::freeListCount; level++)
            {
                // have any free blocks
                if (!Base::freeListStackCtr[level])
                    continue;

                // pop off the top
                const Block& popped = Base::freeListStack[level][--Base::freeListStackCtr[level]];
                Block tmp;
                size_type wastedSpace = Base::calcSubAllocation(tmp,&popped,bytes,alignment);
                // if had a block large enough for us with padding then must be able to allocate
                #ifdef _IRR_DEBUG
                    if (wastedSpace==invalid_address)
                        assert(false);
                #endif // _IRR_DEBUG
                Base::freeSize -= popped.getLength();
                return std::pair<Block,Block>(tmp,popped);
            }

            return std::pair<Block,Block>(Block{invalid_address,invalid_address},Block{invalid_address,invalid_address});
        }
};

}

//! General-purpose allocator, really its like a buddy allocator that supports more sophisticated coalescing
template<typename _size_type, class AllocStrategy = impl::GeneralpurposeAddressAllocatorStrategy<_size_type,false> >
class GeneralpurposeAddressAllocator : public AddressAllocatorBase<GeneralpurposeAddressAllocator<_size_type>,_size_type>, protected AllocStrategy
{
    private:
        typedef AddressAllocatorBase<GeneralpurposeAddressAllocator<_size_type>,_size_type> Base;
        typedef typename AllocStrategy::Block                                               Block;
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        #define DUMMY_DEFAULT_CONSTRUCTOR GeneralpurposeAddressAllocator() noexcept : AllocStrategy(invalid_address,invalid_address) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~GeneralpurposeAddressAllocator() {}

        GeneralpurposeAddressAllocator(void* reservedSpc, size_type addressOffsetToApply, size_type alignOffsetNeeded, size_type maxAllocatableAlignment, size_type bufSz, size_type minBlockSz) noexcept :
                    Base(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment), AllocStrategy(bufSz-Base::alignOffset,minBlockSz)
        {
            // buffer has to be large enough for at least one block of minimum size, buffer has to be smaller than magic value
            assert(bufSz>=Base::alignOffset+AllocStrategy::minBlockSize && AllocStrategy::bufferSize<invalid_address);
            // max free block size (buffer size) must not force the segregated free list to have too many levels
            assert(AllocStrategy::findFreeListInsertIndex(AllocStrategy::bufferSize) < AllocStrategy::maxListLevels);

            reset();
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator even if `supportsNullBuffer==true`
        template<typename... Args>
        GeneralpurposeAddressAllocator(size_type newBuffSz, GeneralpurposeAddressAllocator&& other, void* newReservedSpc, Args&&... args) noexcept :
                    Base(std::move(other),newReservedSpc,std::forward<Args>(args)...),
                    AllocStrategy(newBuffSz-Base::alignOffset,std::move(other),newReservedSpc)
        {
        }

        GeneralpurposeAddressAllocator& operator=(GeneralpurposeAddressAllocator&& other)
        {
            Base::operator=(std::move(other));
            AllocStrategy::operator=(std::move(other));
            return *this;
        }

        //! non-PoT alignments cannot be guaranteed after a resize or move of the backing buffer
        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (alignment>Base::maxRequestableAlignment || bytes==0u)
                return invalid_address;

            bytes = std::max(bytes,AllocStrategy::minBlockSize);
            if (bytes>AllocStrategy::freeSize)
                return invalid_address;

            std::pair<Block,Block> found;
            for (auto i=0u; i<2u; i++)
            {
                found = AllocStrategy::findAndPopSuitableBlock(bytes,alignment);

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
                AllocStrategy::insertFreeBlock(Block{found.first.endOffset,found.second.endOffset});
            if (found.first.startOffset!=found.second.startOffset)
                AllocStrategy::insertFreeBlock(Block{found.second.startOffset,found.first.startOffset});

            return found.first.startOffset+Base::combinedOffset;
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
            bytes = std::max(bytes,AllocStrategy::minBlockSize);
#ifdef _IRR_DEBUG
            // address must have had combinedOffset already applied to it, and allocation must not be outside the buffer
            assert(addr>=Base::combinedOffset && addr+bytes<=AllocStrategy::bufferSize+Base::combinedOffset);
            // sanity check
            assert(AllocStrategy::freeSize+bytes<=AllocStrategy::bufferSize);
#endif // _IRR_DEBUG

            addr -= Base::combinedOffset;
#ifdef _EXTREME_DEBUG
            // double free protection
            assert(!AllocStrategy::is_double_free(addr,bytes));
#endif // _EXTREME_DEBUG
            AllocStrategy::insertFreeBlock(Block{addr,addr+bytes});
        }

        inline void             reset()
        {
            AllocStrategy::swapFreeLists(Base::reservedSpace);
            AllocStrategy::insertFreeBlock(Block{0u,AllocStrategy::bufferSize});
        }

        //! Conservative estimate, max_size() gives largest size we are sure to be able to allocate
        inline size_type        max_size() const noexcept
        {
            const auto maxWastedSpace   = AllocStrategy::getBiggestPossibleFrontWaste(Base::maxRequestableAlignment)+AllocStrategy::minBlockSize;
            const auto lowestLimit      = AllocStrategy::findFreeListSearchIndex(maxWastedSpace);
            for (decltype(AllocStrategy::freeListCount) i=AllocStrategy::freeListCount; i>lowestLimit; i--)
            {
                size_type level = i-1u;
                auto blockCount = AllocStrategy::freeListStackCtr[level];
                if (!blockCount)
                    continue;

                // get first block in the size's free-list, not accurate since there might be bigger blocks further in the list.
                // however because the free-lists are binned by size, this is accurate within a factor of 1.99999999x
                blockCount--;
                auto length = AllocStrategy::freeListStack[level][blockCount].getLength();
                if (length>maxWastedSpace)
                    return length-maxWastedSpace;
            }

            return 0u;
        }

        //! Most allocators do not support e.g. 1-byte allocations
        inline size_type        min_size() const noexcept
        {
            return AllocStrategy::minBlockSize;
        }

        inline size_type        safe_shrink_size(size_type sizeBound, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            size_type retval = get_total_size()-Base::alignOffset;
            if (sizeBound>=retval)
                return Base::safe_shrink_size(sizeBound,newBuffAlignmentWeCanGuarantee);

            if (get_free_size()==0u)
                return Base::safe_shrink_size(retval,newBuffAlignmentWeCanGuarantee);

            //now increase sizeBound by taking into account fragmentation
            retval = defragment();

            return Base::safe_shrink_size(std::max(retval,sizeBound),newBuffAlignmentWeCanGuarantee);
        }


        static inline size_type reserved_size(size_type maxAlignment, size_type bufSz, size_type minBlockSize) noexcept
        {
            size_type reserved = 0u;
            for (size_type i=0u; i<AllocStrategy::findFreeListCount(bufSz,minBlockSize); i++)
                reserved += (bufSz/(minBlockSize<<i)+1u)*size_type(2u);
            return (reserved-2u)*sizeof(Block);
        }
        static inline size_type reserved_size(size_type bufSz, const GeneralpurposeAddressAllocator<_size_type>& other) noexcept
        {
            return reserved_size(bufSz,other.maxRequestableAlignment,other.blockSize);
        }

        inline size_type        get_free_size() const noexcept
        {
            return AllocStrategy::freeSize; // decrement when allocating, increment when freeing
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return AllocStrategy::bufferSize-AllocStrategy::freeSize;
        }
        inline size_type        get_total_size() const noexcept
        {
            return AllocStrategy::bufferSize+Base::alignOffset;
        }

        inline bool                 is_double_free(size_type addr, size_type bytes) const noexcept
        {
            return AllocStrategy::is_double_free(addr-Base::combinedOffset,bytes);
        }
    protected:
        inline size_type        defragment() noexcept
        {
            // TODO: radix sort the whole thing on the block-start value and do a coalesce without `AllocStrategy::findMinimum`
            // also add the blocks in reverse order
            Block* freeListOld[AllocStrategy::maxListLevels];
            const Block* freeListOldEnd[AllocStrategy::maxListLevels];
            for (decltype(AllocStrategy::freeListCount) i=0u; i<AllocStrategy::freeListCount; i++)
            {
                freeListOld[i] = AllocStrategy::freeListStack[i];
                freeListOldEnd[i] = freeListOld[i]+AllocStrategy::freeListStackCtr[i];
                std::sort(freeListOld[i],const_cast<Block*>(freeListOldEnd[i]));
            }

            AllocStrategy::swapFreeLists(Base::reservedSpace);

            // begin the coalesce
            Block lastBlock{0u,0u};
            auto minimum = AllocStrategy::findMinimum(freeListOld,freeListOldEnd);
            while (minimum!=AllocStrategy::freeListCount)
            {
                // find next free block and pop it
                const Block* nextBlock = freeListOld[minimum]++;

                // check if broke continuity
                if (nextBlock->startOffset!=lastBlock.endOffset)
                {
                    // put old on correct free list
                    if (lastBlock.getLength())
                        AllocStrategy::insertFreeBlock(lastBlock);

                    lastBlock.startOffset = nextBlock->startOffset;
                }

                lastBlock.endOffset = nextBlock->endOffset;
                minimum = AllocStrategy::findMinimum(freeListOld,freeListOldEnd);
            }
            #ifdef _IRR_DEBUG
            for (decltype(AllocStrategy::freeListCount) i=0u; i<AllocStrategy::freeListCount; i++)
                assert(freeListOld[i]==freeListOldEnd[i]);
            #endif // _IRR_DEBUG
            // put last block on correct free list
            if (lastBlock.getLength())
            {
                AllocStrategy::insertFreeBlock(lastBlock);
                if (lastBlock.endOffset==AllocStrategy::bufferSize)
                    return lastBlock.startOffset;
            }

            return AllocStrategy::bufferSize;
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

template<typename size_type, class RecursiveLockable>
using GeneralpurposeAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<GeneralpurposeAddressAllocator<size_type>,RecursiveLockable>;

}
}

#endif // __IRR_GENERALPURPOSE_ADDRESS_ALLOCATOR_H_INCLUDED__


