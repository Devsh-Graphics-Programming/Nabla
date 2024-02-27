// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_SUB_ALLOCATED_DESCRIPTOR_SET_H_
#define _NBL_VIDEO_SUB_ALLOCATED_DESCRIPTOR_SET_H

#include "nbl/video/alloc/IBufferAllocator.h"

#include <type_traits>
#include <map>

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
		inline DeferredFreeFunctor(SubAllocatedDescriptorSet* composed, uint32_t binding, size_type count, const value_type* addresses)
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
			m_composed->multi_deallocate(m_binding, m_addresses.size(), m_addresses.data());
		}

		// Takes count of allocations we want to free up as reference, true is returned if
		// the amount of allocations freed was >= allocationsToFreeUp
		// False is returned if there are more allocations to free up
		inline bool operator()(size_type& allocationsToFreeUp)
		{
			auto prevCount = m_addresses.size();
			operator()();
			auto totalFreed = m_addresses.size() - prevCount;

			// This does the same logic as bool operator()(size_type&) on 
			// CAsyncSingleBufferSubAllocator
			bool freedEverything = totalFreed >= allocationsToFreeUp;
		
			if (freedEverything) allocationsToFreeUp = 0u;
			else allocationsToFreeUp -= totalFreed;
			return freedEverything;
		}
	protected:
		SubAllocatedDescriptorSet* m_composed;
		uint32_t m_binding;
		std::vector<value_type> m_addresses;
	};
	using EventHandler = MultiTimelineEventHandlerST<DeferredFreeFunctor>;
protected:
	struct SubAllocDescriptorSetRange {
		EventHandler eventHandler = EventHandler({});
		std::unique_ptr<AddressAllocator> addressAllocator = nullptr;
		std::unique_ptr<ReservedAllocator> reservedAllocator = nullptr;
		size_t reservedSize = 0;

		SubAllocDescriptorSetRange(
			std::unique_ptr<AddressAllocator>&& inAddressAllocator,
			std::unique_ptr<ReservedAllocator>&& inReservedAllocator,
			size_t inReservedSize) :
			eventHandler({}), addressAllocator(std::move(inAddressAllocator)),
			reservedAllocator(std::move(inReservedAllocator)), reservedSize(inReservedSize) {}
		SubAllocDescriptorSetRange() {}

		SubAllocDescriptorSetRange& operator=(SubAllocDescriptorSetRange&& other)
		{
			addressAllocator = std::move(other.addressAllocator);
			reservedAllocator = std::move(other.reservedAllocator);
			reservedSize = other.reservedSize;
			return *this;
		}
	};
	std::map<uint32_t, SubAllocDescriptorSetRange> m_allocatableRanges = {};
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_descriptorSet;
	core::smart_refctd_ptr<video::ILogicalDevice> m_logicalDevice;

	#ifdef _NBL_DEBUG
	std::recursive_mutex stAccessVerfier;
	#endif // _NBL_DEBUG

	constexpr static inline uint32_t MaxDescriptorSetAllocationAlignment = 1u; 
	constexpr static inline uint32_t MinDescriptorSetAllocationSize = 1u;

public:

	// constructors
	inline SubAllocatedDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSet>&& descriptorSet,
		core::smart_refctd_ptr<video::ILogicalDevice>&& logicalDevice) : m_logicalDevice(std::move(logicalDevice))
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
					auto reservedSize = AddressAllocator::reserved_size(MaxDescriptorSetAllocationAlignment, static_cast<size_type>(count), MinDescriptorSetAllocationSize);
					auto reservedAllocator = std::unique_ptr<ReservedAllocator>(new ReservedAllocator());
					auto addressAllocator = std::unique_ptr<AddressAllocator>(new AddressAllocator(
						reservedAllocator->allocate(reservedSize, _NBL_SIMD_ALIGNMENT),
						static_cast<size_type>(0), 0u, MaxDescriptorSetAllocationAlignment, static_cast<size_type>(count),
						MinDescriptorSetAllocationSize
					));

					m_allocatableRanges[binding.data] = SubAllocDescriptorSetRange(std::move(addressAllocator), std::move(reservedAllocator), reservedSize);
				}
			}
		}
		m_descriptorSet = std::move(descriptorSet);
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
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return nullptr;
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

	video::IGPUDescriptorSet* getDescriptorSet() { return m_descriptorSet.get(); }

	//! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
	inline size_type try_multi_allocate(uint32_t binding, size_type count, video::IGPUDescriptorSet::SDescriptorInfo* descriptors, value_type* outAddresses)
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto allocator = getBindingAllocator(binding);

		std::vector<video::IGPUDescriptorSet::SWriteDescriptorSet> writes;
		std::vector<video::IGPUDescriptorSet::SDescriptorInfo> infos;
		writes.reserve(count);
		infos.reserve(count);

		size_type unallocatedSize = 0u;
		for (size_type i=0; i<count; i++)
		{
			if (outAddresses[i]!=AddressAllocator::invalid_address)
				continue;

			outAddresses[i] = allocator->alloc_addr(1,1);
			if (outAddresses[i] == AddressAllocator::invalid_address)
			{
				unallocatedSize = count - i;
				break;
			}

			auto& descriptor = descriptors[i];
			
			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			{
				write.dstSet = m_descriptorSet.get();
				write.binding = binding;
				write.arrayElement = outAddresses[i];
				write.count = 1u;
				// descriptors could be a const pointer, but the problem is that this pointer in
				// SWriteDescriptorSet.info isn't const
				// can we change it?
				write.info = &descriptor;
			}
			infos.push_back(descriptor);
			writes.push_back(write);
		}

		m_logicalDevice->updateDescriptorSets(writes, {});
		return unallocatedSize;
	}

	template<class Clock=typename std::chrono::steady_clock>
	inline size_type multi_allocate(const std::chrono::time_point<Clock>& maxWaitPoint, uint32_t binding, size_type count, video::IGPUDescriptorSet::SDescriptorInfo* descriptors, value_type* outAddresses) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto range = m_allocatableRanges.find(binding);
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return count;

		auto& eventHandler = range->second.eventHandler;

		// try allocate once
		size_type unallocatedSize = try_multi_allocate(binding, count, descriptors, outAddresses);
		if (!unallocatedSize)
			return 0u;

		// then try to wait at least once and allocate
		do
		{
			eventHandler.wait(maxWaitPoint, unallocatedSize);

			unallocatedSize = try_multi_allocate(binding, unallocatedSize, &descriptors[count - unallocatedSize], &outAddresses[count - unallocatedSize]);
			if (!unallocatedSize)
				return 0u;
		} while(Clock::now()<maxWaitPoint);

		return unallocatedSize;
	}

	inline size_type multi_allocate(uint32_t binding, size_type count, video::IGPUDescriptorSet::SDescriptorInfo* descriptors, value_type* outAddresses) noexcept
	{
		auto range = m_allocatableRanges.find(binding);
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return count;

		return multi_allocate(TimelineEventHandlerBase::default_wait(), binding, count, descriptors, outAddresses);
	}

	inline void multi_deallocate(uint32_t binding, size_type count, const size_type* addr)
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto allocator = getBindingAllocator(binding);
		if (!allocator)
			return;
		for (size_type i=0; i<count; i++)
		{
			if (addr[i]==AddressAllocator::invalid_address)
				continue;

			allocator->free_addr(addr[i],1);
			// TODO: should also write something to the descriptor sets
		}
	}

	inline void multi_deallocate(uint32_t binding, const ISemaphore::SWaitInfo& futureWait, DeferredFreeFunctor&& functor) noexcept
	{
		auto range = m_allocatableRanges.find(binding);
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return;

		auto& eventHandler = range->second.eventHandler;
		auto debugGuard = stAccessVerifyDebugGuard();
		eventHandler.latch(futureWait,std::move(functor));
	}

	inline void multi_deallocate(uint32_t binding, size_type count, const value_type* addr, const ISemaphore::SWaitInfo& futureWait) noexcept
	{
		if (futureWait.semaphore)
			multi_deallocate(binding, futureWait, DeferredFreeFunctor(this, binding, count, addr));
		else
			multi_deallocate(binding, count, addr);
	}
	//! Returns free events still outstanding
	inline uint32_t cull_frees() noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();
		uint32_t frees = 0;
		for (uint32_t i = 0; i < m_allocatableRanges.size(); i++)
		{
			auto& it = m_allocatableRanges[i];
			frees += it.eventHandler.poll().eventsLeft;
		}
		return frees;
	}
};

}

#endif
