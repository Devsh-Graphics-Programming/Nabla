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

	class DeferredFreeFunctor
	{
	public:
		inline DeferredFreeFunctor(SubAllocatedDescriptorSet* composed, uint32_t binding, size_type count, value_type* addresses)
			: m_addresses(addresses, addresses + count), m_binding(binding), m_composed(composed)
		{
		}

		// Just does the de-allocation
		inline void operator()()
		{
			// isn't assert already debug-only?
			#ifdef _NBL_DEBUG
			assert(m_composed);
			#endif // _NBL_DEBUG
			m_composed->multi_deallocate(m_binding, m_addresses.size(), &m_addresses[0]);
		}

		// Takes count of allocations we want to free up as reference, true is returned if
		// the amount of allocations freed was >= allocationsToFreeUp
		// False is returned if there are more allocations to free up
		inline bool operator()(size_type allocationsToFreeUp)
		{
			auto prevCount = m_addresses.size();
			operator()();
			auto totalFreed = m_addresses.size() - prevCount;

			// This does the same logic as bool operator()(size_type&) on 
			// CAsyncSingleBufferSubAllocator
			return totalFreed >= allocationsToFreeUp;
		}
	protected:
		SubAllocatedDescriptorSet* m_composed;
		uint32_t m_binding;
		std::vector<value_type> m_addresses;
	};
protected:
	struct SubAllocDescriptorSetRange {
		std::shared_ptr<AddressAllocator> addressAllocator;
		std::shared_ptr<ReservedAllocator> reservedAllocator;
		size_t reservedSize;
	};
	MultiTimelineEventHandlerST<DeferredFreeFunctor> eventHandler;
	std::map<uint32_t, SubAllocDescriptorSetRange> m_allocatableRanges = {};
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_descriptorSet;

	#ifdef _NBL_DEBUG
	std::recursive_mutex stAccessVerfier;
	#endif // _NBL_DEBUG

	constexpr static inline uint32_t MaxDescriptorSetAllocationAlignment = 1u; 
	constexpr static inline uint32_t MinDescriptorSetAllocationSize = 1u;

public:

	// constructors
	template<typename... Args>
	inline SubAllocatedDescriptorSet(video::IGPUDescriptorSet* descriptorSet)
	{
		auto layout = descriptorSet->getLayout();
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
					range.reservedSize = AddressAllocator::reserved_size(MaxDescriptorSetAllocationAlignment, static_cast<size_type>(count), MinDescriptorSetAllocationSize);
					range.reservedAllocator = std::shared_ptr<ReservedAllocator>(new ReservedAllocator());
					range.addressAllocator = std::shared_ptr<AddressAllocator>(new AddressAllocator(
						range.reservedAllocator->allocate(range.reservedSize, _NBL_SIMD_ALIGNMENT),
						static_cast<size_type>(0), 0u, MaxDescriptorSetAllocationAlignment, static_cast<size_type>(count),
						MinDescriptorSetAllocationSize
					));
					m_allocatableRanges.emplace(binding.data, range);
				}
			}
		}
		m_descriptorSet = core::smart_refctd_ptr(descriptorSet);
	}

	~SubAllocatedDescriptorSet()
	{
		for (uint32_t i = 0; i < m_allocatableRanges.size(); i++)
		{
			auto& range = m_allocatableRanges[i];
			if (range.reservedSize == 0)
				continue;
			auto ptr = reinterpret_cast<const uint8_t*>(core::address_allocator_traits<AddressAllocator>::getReservedSpacePtr(*range.addressAllocator));
			range.addressAllocator = nullptr;
			range.reservedAllocator->deallocate(const_cast<uint8_t*>(ptr), range.reservedSize);
		}
	}

	// whether that binding index can be sub-allocated
	bool isBindingAllocatable(uint32_t binding) { return m_allocatableRanges.find(binding) != m_allocatableRanges.end(); }

	AddressAllocator* getBindingAllocator(uint32_t binding) 
	{ 
		auto range = m_allocatableRanges.find(binding);
		assert(range != m_allocatableRanges.end());// Check if this binding has an allocator
		return range->second.addressAllocator.get(); 
	}

	// main methods

#ifdef _NBL_DEBUG
	std::unique_lock<std::recursive_mutex> stAccessVerifyDebugGuard()
	{
		std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
		assert(tLock.owns_lock());
		return tLock;
	}
#else
	bool stAccessVerifyDebugGuard() { return false; }
#endif

	//! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
	inline void multi_allocate(uint32_t binding, size_type count, value_type* outAddresses)
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto allocator = getBindingAllocator(binding);
		for (size_type i=0; i<count; i++)
		{
			if (outAddresses[i]!=AddressAllocator::invalid_address)
				continue;

			outAddresses[i] = allocator->alloc_addr(1,1);
			// TODO: should also write something to the descriptor set (or probably leave that to the caller?)
		}
	}
	inline void multi_deallocate(uint32_t binding, size_type count, const size_type* addr)
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto allocator = getBindingAllocator(binding);
		for (size_type i=0; i<count; i++)
		{
			if (addr[i]==AddressAllocator::invalid_address)
				continue;

			allocator->free_addr(addr[i],1);
			// TODO: should also write something to the descriptor sets
		}
	}
	//!
	inline void multi_deallocate(const ISemaphore::SWaitInfo& futureWait, DeferredFreeFunctor&& functor) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();
		eventHandler.latch(futureWait,std::move(functor));
	}
	// TODO: improve signature of this function in the future
	template<typename T=core::IReferenceCounted>
	inline void multi_deallocate(uint32_t binding, uint32_t count, const value_type* addr, const ISemaphore::SWaitInfo& futureWait) noexcept
	{
		if (futureWait.semaphore)
			multi_deallocate(futureWait, DeferredFreeFunctor(&this, binding, count, addr));
		else
			multi_deallocate(binding, count, addr);
	}
	//! Returns free events still outstanding
	inline uint32_t cull_frees() noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();
		return eventHandler.poll().eventsLeft;
	}
};

}

#endif
