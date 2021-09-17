// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_PROPERTY_POOL_H_INCLUDED__
#define __NBL_VIDEO_I_PROPERTY_POOL_H_INCLUDED__


#include "nbl/asset/asset.h"

#include "nbl/video/IGPUBuffer.h"


namespace nbl::video
{


// property pool is inherently single threaded
class IPropertyPool : public core::IReferenceCounted
{
	public:
		using PropertyAddressAllocator = core::PoolAddressAllocatorST<uint32_t>;

        static inline constexpr auto invalid = PropertyAddressAllocator::invalid_address;

		//
        virtual const asset::SBufferRange<IGPUBuffer>& getPropertyMemoryBlock(uint32_t ix) const =0;

		//
		virtual uint32_t getPropertyCount() const =0;
		virtual uint32_t getPropertySize(uint32_t ix) const =0;

        //
        inline bool isContiguous() const {return m_indexToAddr;}

        //
        inline uint32_t getAllocated() const
        {
            return indexAllocator.get_allocated_size();
        }

        //
        inline uint32_t getFree() const
        {
            return indexAllocator.get_free_size();
        }

        //
        inline uint32_t getCapacity() const
        {
            // special case allows us to use `get_total_size`, because the pool allocator has no added offsets
            return indexAllocator.get_total_size();
        }

        // allocate, indices need to be pre-initialized to `invalid`
        inline bool allocateProperties(uint32_t* outIndicesBegin, uint32_t* outIndicesEnd)
        {
            constexpr uint32_t unit = 1u;
            uint32_t head = getAllocated();
            for (auto it=outIndicesBegin; it!=outIndicesEnd; it++)
            {
                auto& index = *it;
                if (index!=invalid)
                    continue;

                index = indexAllocator.alloc_addr(unit,unit);
                if (index==invalid)
                    return false;
                
                if (isContiguous())
                {
                    assert(m_indexToAddr[index]==invalid);
                    assert(m_addrToIndex[head]==invalid);
                    m_indexToAddr[index] = head;
                    m_addrToIndex[head++] = index;
                }
            }
            return true;
        }

        // WARNING: For contiguous pools, YOU NEED TO ISSUE A TransferRequest WITH `tailMoveDest` BEFORE YOU ALLOCATE OR FREE ANY MORE PROPS!
        // This function does not move the properties' contents, for contiguous pools it means you WILL loose the tail's data if you dont do this!
        // Returns how many elements need to be transferred from the tail to fill the gaps after deallocation in a contiguous pool.
        [[nodiscard]] inline uint32_t freeProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd, uint32_t* srcAddresses, uint32_t* dstAddresses)
        {
            const uint32_t oldHead = getAllocated();
            constexpr uint32_t unit = 1u;
            for (auto it=indicesBegin; it!=indicesEnd; it++)
            {
                if (*it==invalid)
                    continue;
                indexAllocator.free_addr(*it,unit);
            }
            const uint32_t head = getAllocated();
            const uint32_t removedCount = oldHead-head;
            // no point trying to move anything if we've freed nothing or everything
            if (isContiguous() && head!=oldHead && head!=0u)
            {
                if (!srcAddresses || !dstAddresses)
                {
                    assert(false);
                    exit(0xdeadbeefu);
                }
                auto gapIt = dstAddresses;
                for (auto it=indicesBegin; it!=indicesEnd; it++)
                {
                    const auto index = *it;
                    if (index==invalid)
                        continue;
                    auto& addr = m_indexToAddr[index];
                    if (addr<head) // overwrite if address in live range
                        *(gapIt++) = addr;
                    else // mark as dead if outside
                        m_addrToIndex[addr] = invalid;
                    // index doesn't map to any address anymore
                    addr = invalid;
                }
                gapIt = dstAddresses; // rewind
                for (auto a=head; a<oldHead; a++)
                {
                    auto& index = m_addrToIndex[a];
                    // marked as dead by previous pass
                    if (index==invalid)
                        continue;
                    // if not dead we need to move
                    *(srcAddresses++) = a;
                    const auto freeAddr = *(gapIt++);
                    m_addrToIndex[freeAddr] = index;
                    m_indexToAddr[index] = freeAddr;
                    index = invalid;
                }
                return (gapIt-dstAddresses);
            }
            return 0u;
        }
        inline uint32_t freeProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd)
        {
            return freeProperties(indicesBegin,indicesEnd,nullptr,nullptr);
        }

        //
        template<typename ConstIterator, typename Iterator>
        inline void indicesToAddresses(ConstIterator begin, ConstIterator end, Iterator dst) const
        {
            if (isContiguous())
            {
                for (auto it=begin; it!=end; it++)
                    *(dst++) = m_indexToAddr[*it];
            }
            else if (begin!=dst)
                std::copy(begin,end,dst);
        }
        template<typename ConstIterator, typename Iterator>
        inline void addressesToIndices(ConstIterator begin, ConstIterator end, Iterator dst) const
        {
            if (isContiguous())
            {
                for (auto it=begin; it!=end; it++)
                    *(dst++) = m_addrToIndex[*it];
            }
            else if (begin!=dst)
                std::copy(begin,end,dst);
        }

        //
        inline void freeAllProperties()
        {
            if (isContiguous())
            {
                std::fill_n(m_indexToAddr,getCapacity(),invalid);
                std::fill_n(m_addrToIndex,getAllocated(),invalid);
            }
            indexAllocator.reset();
        }
        
        //
        static PropertyAddressAllocator::size_type getReservedSize(uint32_t capacity, bool contiguous=false);

    protected:
        IPropertyPool(uint32_t capacity, void* reserved, bool contiguous=false);
        virtual ~IPropertyPool() {}

        static bool validateBlocks(const ILogicalDevice* device, const uint32_t propertyCount, const size_t* propertySizes, const uint32_t capacity, const asset::SBufferRange<IGPUBuffer>* _memoryBlocks);

        PropertyAddressAllocator indexAllocator;
        uint32_t* m_indexToAddr;
        uint32_t* m_addrToIndex;
};


}

#endif