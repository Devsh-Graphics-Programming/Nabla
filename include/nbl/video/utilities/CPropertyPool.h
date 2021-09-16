// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_H_INCLUDED__

#include "nbl/video/utilities/IPropertyPool.h"


namespace nbl::video
{

    
template<template<class...> class allocator=core::allocator, typename... Properties>
class CPropertyPool final : public IPropertyPool
{
        using this_t = CPropertyPool<allocator,Properties...>;

        static auto propertyCombinedSize()
        {
            return (sizeof(Properties) + ...);
        }

	public:
        static inline constexpr auto PropertyCount = sizeof...(Properties);
        static inline constexpr std::array<size_t,PropertyCount> PropertySizes = {sizeof(Properties)...};

        //
        static inline uint32_t calcApproximateCapacity(const asset::SBufferRange<IGPUBuffer>* _memoryBlocks)
        {
            size_t capacity = ~0ull;
            for (auto i=1u; i<PropertyCount; i++)
            {
                const auto bufcap = _memoryBlocks[0].size/PropertySizes[0];
                if (bufcap<capacity)
                    capacity = bufcap;
            }
            return core::min<size_t>(IPropertyPool::invalid,capacity);
        }

        // easy dont care creation
        static inline core::smart_refctd_ptr<this_t> create(ILogicalDevice* device, const uint32_t capacity, const bool contiguous = false)
        {
            asset::SBufferRange<video::IGPUBuffer> blocks[PropertyCount];
            for (auto i=0u; i<PropertyCount; i++)
            {
                blocks[i].offset = 0ull;
                blocks[i].size = capacity*PropertySizes[i];
                blocks[i].buffer = device->createDeviceLocalGPUBufferOnDedMem(blocks[i].size);
            }
            return create(device,blocks,capacity,contiguous,allocator<uint8_t>());
        }
        // you can either construct the pool with explicit capacity or have it deduced from the memory blocks you pass
		static inline core::smart_refctd_ptr<this_t> create(const ILogicalDevice* device, const asset::SBufferRange<IGPUBuffer>* _memoryBlocks, uint32_t capacity=0u, const bool contiguous=false, allocator<uint8_t>&& alloc = allocator<uint8_t>())
		{
            if (!capacity)
                capacity = calcApproximateCapacity(_memoryBlocks);
			const auto reservedSize = getReservedSize(capacity,contiguous);
			auto reserved = std::allocator_traits<allocator<uint8_t>>::allocate(alloc,reservedSize);
			if (!reserved)
				return nullptr;

			auto retval = create(device,_memoryBlocks,capacity,reserved,contiguous,std::move(alloc));
			if (!retval)
				std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,reserved,reservedSize);

			return retval;
		}
		// if this method fails to create the pool, the callee must free the reserved memory themselves, also the reserved pointer must be compatible with the allocator so it can free it
        static inline core::smart_refctd_ptr<this_t> create(const ILogicalDevice* device, const asset::SBufferRange<IGPUBuffer>* _memoryBlocks, const uint32_t capacity, void* reserved, const bool contiguous=false, allocator<uint8_t>&& alloc=allocator<uint8_t>())
        {
            if (!IPropertyPool::validateBlocks(device,PropertyCount,PropertySizes.data(),capacity,_memoryBlocks))
                return nullptr;
            if (!reserved || !capacity)
                return nullptr;

			auto* pool = new CPropertyPool(_memoryBlocks,capacity,reserved,contiguous,std::move(alloc));
            return core::smart_refctd_ptr<CPropertyPool>(pool,core::dont_grab);
        }


        //
        const asset::SBufferRange<IGPUBuffer>& getPropertyMemoryBlock(uint32_t ix) const override {return m_memoryBlocks[ix];}

		//
		uint32_t getPropertyCount() const override {return PropertyCount;}
		uint32_t getPropertySize(uint32_t ix) const override {return static_cast<uint32_t>(PropertySizes[ix]);}

	protected:
        CPropertyPool(const asset::SBufferRange<IGPUBuffer>* _memoryBlocks, const uint32_t capacity, void* _reserved, bool contiguous, allocator<uint8_t>&& _alloc)
            : IPropertyPool(capacity,_reserved,contiguous), alloc(std::move(_alloc)), reserved(_reserved)
        {
            std::copy_n(_memoryBlocks,getPropertyCount(),m_memoryBlocks);
        }

        ~CPropertyPool()
        {
            std::allocator_traits<allocator<uint8_t>>::deallocate(alloc,reinterpret_cast<uint8_t*>(reserved),getReservedSize(getCapacity()));
        }


        allocator<uint8_t> alloc;
        asset::SBufferRange<IGPUBuffer> m_memoryBlocks[PropertyCount];
		void* reserved;
};


}

#endif