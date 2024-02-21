// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_SUB_ALLOCATED_DESCRIPTOR_SET_H_
#define _NBL_VIDEO_SUB_ALLOCATED_DESCRIPTOR_SET_H

#include "nbl/video/alloc/IBufferAllocator.h"

#include <type_traits>

namespace nbl::video
{

class SubAllocatedDescriptorSet : public core::IReferenceCounted 
{
public:
	// address allocator gives offsets
	// reserved allocator allocates memory to keep the address allocator state inside
	using AddressAllocator = core::PoolAddressAllocator<uint32_t>;
	using ReservedAllocator = core::allocator<uint8_t>;
	using size_type = typename AddressAllocator::size_type;
	using value_type = typename AddressAllocator::size_type;
	static constexpr value_type invalid_value = AddressAllocator::invalid_address;

protected:
	struct SubAllocDescriptorSetRange {
		std::shared_ptr<AddressAllocator> addressAllocator;
		std::shared_ptr<ReservedAllocator> reservedAllocator;
		size_t reservedSize;
	};
	std::unordered_map<uint32_t, SubAllocDescriptorSetRange> m_allocatableRanges = {};


public:
	// constructors
	template<typename... Args>
	inline SubAllocatedDescriptorSet(video::IGPUDescriptorSetLayout* layout, const value_type maxAllocatableAlignment, Args&&... args)
	{
		for (uint32_t descriptorType = 0; descriptorType < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); descriptorType++)
		{
			auto descType = static_cast<asset::IDescriptor::E_TYPE>(descriptorType);
			auto& redirect = layout->getDescriptorRedirect(descType);

			for (uint32_t i = 0; i < redirect.getBindingCount(); i++)
			{
				auto binding = redirect.getBinding(i);
				auto storageIndex = redirect.findBindingStorageIndex(binding);

				auto count = redirect.getCount(storageIndex);
				auto flags = redirect.getCreateFlags(storageIndex);

				// Only bindings with these flags will be allocatable
				if (flags.hasFlags(core::bitflag(IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT)
					| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT
					| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_PARTIALLY_BOUND_BIT))
				{
					SubAllocDescriptorSetRange range;
					range.reservedSize = AddressAllocator::reserved_size(maxAllocatableAlignment, static_cast<size_type>(count), args...);
					range.reservedAllocator = std::shared_ptr<ReservedAllocator>(new ReservedAllocator());
					range.addressAllocator = std::shared_ptr<AddressAllocator>(new AddressAllocator(
						range.reservedAllocator->allocate(range.reservedSize, _NBL_SIMD_ALIGNMENT),
						static_cast<size_type>(0), 0u, maxAllocatableAlignment, static_cast<size_type>(count), std::forward<Args>(args)...
					));
					m_allocatableRanges.emplace(binding.data, range);
				}
			}
		}
	}

	~SubAllocatedDescriptorSet()
	{
		for (uint32_t i = 0; i < m_allocatableRanges.size(); i++)
		{
			auto& range = m_allocatableRanges[i];
			if (range.reservedSize == 0)
				continue;

			auto ptr = reinterpret_cast<const uint8_t*>(core::address_allocator_traits<AddressAllocator>::getReservedSpacePtr(*range.addressAllocator));
			range.reservedAllocator->deallocate(const_cast<uint8_t*>(ptr), range.reservedSize);
		}
	}

	// amount of bindings in the descriptor set layout used
	uint32_t getLayoutBindingCount() { return m_allocatableRanges.size(); }

	// whether that binding index can be sub-allocated
	bool isBindingAllocatable(uint32_t binding) 
	{ 
		return m_allocatableRanges.find(binding) != m_allocatableRanges.end(); 
	}

	AddressAllocator& getBindingAllocator(uint32_t binding) 
	{ 
		auto range = m_allocatableRanges.find(binding);
		assert(range != m_allocatableRanges.end());// Check if this binding has an allocator
		return *range->second.addressAllocator; 
	}

	// main methods

	//! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
	inline void multi_allocate(uint32_t binding, uint32_t count, value_type* outAddresses)
	{
		auto& allocator = getBindingAllocator(binding);
		for (uint32_t i=0; i<count; i++)
		{
			if (outAddresses[i]!=AddressAllocator::invalid_address)
				continue;

			outAddresses[i] = allocator.alloc_addr(1,1);
		}
	}
	inline void multi_deallocate(uint32_t binding, uint32_t count, const size_type* addr)
	{
		auto& allocator = getBindingAllocator(binding);
		for (uint32_t i=0; i<count; i++)
		{
			if (addr[i]==AddressAllocator::invalid_address)
				continue;

			allocator.free_addr(addr[i],1);
		}
	}
};

}

#endif
