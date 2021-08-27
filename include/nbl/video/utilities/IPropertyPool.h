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

        _NBL_STATIC_INLINE_CONSTEXPR auto invalid_index = PropertyAddressAllocator::invalid_address;

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

        // allocate, indices need to be pre-initialized to `invalid_index`
        inline bool allocateProperties(uint32_t* outIndicesBegin, uint32_t* outIndicesEnd)
        {
            constexpr uint32_t unit = 1u;
            uint32_t head = getAllocated();
            for (auto it=outIndicesBegin; it!=outIndicesEnd; it++)
            {
                auto& index = *it;
                if (index!=invalid_index)
                    continue;

                index = indexAllocator.alloc_addr(unit,unit);
                if (index==invalid_index)
                    return false;
                
                if (isContiguous())
                {
                    assert(m_indexToAddr[index]==invalid_index);
                    assert(m_addrToIndex[head]==invalid_index);
                    m_indexToAddr[index] = head;
                    m_addrToIndex[head++] = index;
                }
            }
            return true;
        }

        // WARNING: For contiguous pools, YOU NEED TO ISSUE A TransferRequest WITH `tailMoveDest` BEFORE YOU ALLOCATE OR FREE ANY MORE PROPS!
        // This function does not move the properties' contents, for contiguous pools it means you WILL loose the tail's data otherwise!
        inline uint32_t freeProperties(uint32_t* indicesBegin, uint32_t* indicesEnd, uint32_t* &tailMoveDest)
        {
            const uint32_t oldHead = getAllocated();
            constexpr uint32_t unit = 1u;
            for (auto it=indicesBegin; it!=indicesEnd; it++)
            {
                if (*it==invalid_index)
                    continue;
                indexAllocator.free_addr(*it,unit);
            }
            const uint32_t head = getAllocated();
            const uint32_t removedCount = oldHead-head;
            if (isContiguous())
            {
                if (!tailMoveDest)
                {
                    assert(false);
                    __debugbreak();
                    exit(0xdeadbeefu);
                }
                auto gapIt = indicesBegin; // reuse as temporary storage
                for (auto it=indicesBegin; it!=indicesEnd; it++)
                {
                    if (*it==invalid_index)
                        continue;
                    auto& addr = m_indexToAddr[*it];
                    if (addr<head) // overwrite if address in live range
                        *(gapIt++) = addr;
                    else // mark as dead if outside
                        m_addrToIndex[addr] = invalid_index;
                    // index doesn't map to any address anymore
                    addr = invalid_index;
                }
                gapIt = indicesBegin; // rewind the list of gaps
                for (auto a=oldHead; a<head; a++)
                {
                    auto& index = m_addrToIndex[a];
                    *(tailMoveDest++) = index; // think about whether to add a direct address feeding path to TransferRequest
                    if (index==invalid_index)
                        continue;
                    // if not dead we need to move
                    m_addrToIndex[*gapIt] = index;
                    m_indexToAddr[index] = *(gapIt++);
                    index = invalid_index;
                }
                // then we just do equivalent of
                //for (auto i=0u; i<(srcIt-src); i++)
                    //data[dst[i]] = data[src[i]];
            }
            std::fill(indicesBegin,indicesEnd,invalid_index); // only in debug, or?
            return removedCount;
        }
        inline uint32_t freeProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd)
        {
            uint32_t* dummy = nullptr;
            return freeProperties(const_cast<uint32_t*>(indicesBegin),const_cast<uint32_t*>(indicesEnd),dummy);
        }

        //
        inline uint32_t indexToAddress(const uint32_t index) const
        {
            if (isContiguous())
                return m_indexToAddr[index];
            return index;
        }
        inline uint32_t addressToIndex(const uint32_t addr) const
        {
            if (isContiguous())
                return m_addrToIndex[addr];
            return addr;
        }

        //
        inline void freeAllProperties()
        {
            if (isContiguous())
            {
                std::fill_n(m_indexToAddr,getCapacity(),invalid_index);
                std::fill_n(m_addrToIndex,getAllocated(),invalid_index);
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