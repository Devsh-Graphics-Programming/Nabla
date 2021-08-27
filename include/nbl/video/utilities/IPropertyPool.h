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

        // TODO: return how to copy the tail around
        inline void freeProperties(const uint32_t* indicesBegin, const uint32_t* indicesEnd)
        {
            uint32_t head = getAllocated();

            constexpr uint32_t unit = 1u;
            for (auto it=indicesBegin; it!=indicesEnd; it++)
            {
                auto& index = *it;
                if (index==invalid_index)
                    continue;

                indexAllocator.free_addr(index,unit);
                if (isContiguous())
                {
                    assert(head!=0u);

                    auto& addr = m_indexToAddr[index];
                    auto& lastIx = m_addrToIndex[--head];
                    assert(addr!=invalid_index&&lastIx!=invalid_index);
                    m_indexToAddr[lastIx] = addr;
                    m_addrToIndex[addr] = lastIx;
                    lastIx = invalid_index;
                    addr = invalid_index;
                }
            }
            /* TODO: figure out how to schedule a copy
            for (auto addr=head; addr<oldHead; addr++)
            {
                auto changedIx = m_addrToIndex[addr];
                auto newAddr = m_indexToAddr[changedIx];
                data[newAddr] = data[addr];
            }
            */
        }

        //
        inline uint32_t indexToAddress(const uint32_t index)
        {
            if (isContiguous())
                return m_indexToAddr[index];
            return index;
        }
        inline uint32_t addressToIndex(const uint32_t addr)
        {
            if (isContiguous())
                return m_addrToIndex[addr];
            return addr;
        }

        //
        inline void freeAllProperties()
        {
            // a little trick to reset the arrays to invalid values if we're going to check them with asserts
            bool clearBimap = false;
            assert(clearBimap=isContiguous());
            if (clearBimap)
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