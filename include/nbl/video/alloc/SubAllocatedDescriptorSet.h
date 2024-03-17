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
			: m_addresses(std::move(core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<value_type>>(count))), 
			  m_binding(binding), m_composed(composed)
		{
			memcpy(m_addresses->data(), addresses, count * sizeof(value_type));
		}
		inline DeferredFreeFunctor(DeferredFreeFunctor&& other) 
		{
			operator=(std::move(other));
		}

		//
		inline auto getWorstCaseCount() const {return m_addresses->size();}

		// Just does the de-allocation, note that the parameter is a reference
		inline void operator()(IGPUDescriptorSet::SDropDescriptorSet* &outNullify)
		{
			#ifdef _NBL_DEBUG
			assert(m_composed);
			#endif // _NBL_DEBUG
			outNullify = m_composed->multi_deallocate(outNullify, m_binding, m_addresses->size(), m_addresses->data());
			m_composed->m_totalDeferredFrees -= getWorstCaseCount();
		}

		DeferredFreeFunctor(const DeferredFreeFunctor& other) = delete;
		DeferredFreeFunctor& operator=(const DeferredFreeFunctor& other) = delete;
		inline DeferredFreeFunctor& operator=(DeferredFreeFunctor&& other)
		{
			m_composed = other.m_composed;
			m_addresses = other.m_addresses;
			m_binding = other.m_binding;

			// Nullifying other
			other.m_composed = nullptr;
			other.m_addresses = nullptr;
			other.m_binding = 0;
			return *this;
		}

		// This is needed for the destructor of TimelineEventHandlerST
		// Don't call this directly
		// TODO: Find a workaround for this
		inline void operator()()
		{
			assert(false); // This should not be called, timeline needs to be drained before destructor
			// core::vector<IGPUDescriptorSet::SDropDescriptorSet> nulls(m_addresses->size());
			// auto ptr = nulls.data();
			// operator()(ptr);
			// auto size = ptr - nulls.data();
			// m_composed->m_logicalDevice->nullifyDescriptors({nulls.data(),size_type(size)});
		}

		// Takes count of allocations we want to free up as reference, true is returned if
		// the amount of allocations freed was >= allocationsToFreeUp
		// False is returned if there are more allocations to free up
		inline bool operator()(size_type& allocationsToFreeUp, IGPUDescriptorSet::SDropDescriptorSet* &outNullify)
		{
			auto prevNullify = outNullify;
			operator()(outNullify);
			auto totalFreed = outNullify-prevNullify;

			// This does the same logic as bool operator()(size_type&) on 
			// CAsyncSingleBufferSubAllocator
			bool freedEverything = totalFreed >= allocationsToFreeUp;
		
			if (freedEverything) allocationsToFreeUp = 0u;
			else allocationsToFreeUp -= totalFreed;
			return freedEverything;
		}
	protected:
		core::smart_refctd_dynamic_array<value_type> m_addresses;
		SubAllocatedDescriptorSet* m_composed; // TODO: shouldn't be called `composed`, maybe `parent` or something
		uint32_t m_binding;
	};
	using EventHandler = MultiTimelineEventHandlerST<DeferredFreeFunctor>;
protected:
	struct SubAllocDescriptorSetRange {
		std::unique_ptr<EventHandler> eventHandler = nullptr;
		std::unique_ptr<AddressAllocator> addressAllocator = nullptr;
		std::unique_ptr<ReservedAllocator> reservedAllocator = nullptr;
		size_t reservedSize = 0;
		asset::IDescriptor::E_TYPE descriptorType = asset::IDescriptor::E_TYPE::ET_COUNT;

		SubAllocDescriptorSetRange(
			std::unique_ptr<EventHandler>&& inEventHandler,
			std::unique_ptr<AddressAllocator>&& inAddressAllocator,
			std::unique_ptr<ReservedAllocator>&& inReservedAllocator,
			size_t inReservedSize,
			asset::IDescriptor::E_TYPE inDescriptorType) :
			eventHandler(std::move(inEventHandler)), addressAllocator(std::move(inAddressAllocator)),
			reservedAllocator(std::move(inReservedAllocator)), 
			reservedSize(inReservedSize),
			descriptorType(inDescriptorType) {}
		SubAllocDescriptorSetRange() {}

		SubAllocDescriptorSetRange(const SubAllocDescriptorSetRange& other) = delete;
		SubAllocDescriptorSetRange& operator=(const SubAllocDescriptorSetRange& other) = delete;

		SubAllocDescriptorSetRange& operator=(SubAllocDescriptorSetRange&& other)
		{
			eventHandler = std::move(other.eventHandler);
			addressAllocator = std::move(other.addressAllocator);
			reservedAllocator = std::move(other.reservedAllocator);
			reservedSize = other.reservedSize;
			descriptorType = other.descriptorType;

			// Nullify other
			other.eventHandler = nullptr;
			other.addressAllocator = nullptr;
			other.reservedAllocator = nullptr;
			other.reservedSize = 0u;
			other.descriptorType = asset::IDescriptor::E_TYPE::ET_COUNT;
			return *this;
		}
	};
	std::map<uint32_t, SubAllocDescriptorSetRange> m_allocatableRanges = {};
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_descriptorSet;
	core::smart_refctd_ptr<video::ILogicalDevice> m_logicalDevice;
	value_type m_totalDeferredFrees = 0;

	#ifdef _NBL_DEBUG
	std::recursive_mutex stAccessVerfier;
	#endif // _NBL_DEBUG

	constexpr static inline uint32_t MaxDescriptorSetAllocationAlignment = 1u; 
	constexpr static inline uint32_t MinDescriptorSetAllocationSize = 1u;

public:

	// constructors
	inline SubAllocatedDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSet>&& descriptorSet,
		core::smart_refctd_ptr<video::ILogicalDevice>&& logicalDevice)
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
					auto eventHandler = std::unique_ptr<EventHandler>(new EventHandler(core::smart_refctd_ptr<ILogicalDevice>(logicalDevice)));

					m_allocatableRanges[binding.data] = SubAllocDescriptorSetRange(std::move(eventHandler), std::move(addressAllocator), std::move(reservedAllocator), reservedSize, descType);
					assert(m_allocatableRanges[binding.data].eventHandler->getLogicalDevice());
				}
			}
		}
		m_descriptorSet = std::move(descriptorSet);
		m_logicalDevice = std::move(logicalDevice);
	}

	inline ~SubAllocatedDescriptorSet()
	{
		uint32_t remainingFrees;
		do {
			remainingFrees = cull_frees();
		} while (remainingFrees > 0);

		for (uint32_t i = 0; i < m_allocatableRanges.size(); i++)
		{
			auto& range = m_allocatableRanges[i];
			if (range.reservedSize == 0)
				continue;
			assert(range.eventHandler->getTimelines().size() == 0);
			auto ptr = reinterpret_cast<const uint8_t*>(core::address_allocator_traits<AddressAllocator>::getReservedSpacePtr(*range.addressAllocator));
			range.addressAllocator = nullptr;
			range.reservedAllocator->deallocate(const_cast<uint8_t*>(ptr), range.reservedSize);
		}
	}

	// whether that binding index can be sub-allocated
	inline bool isBindingAllocatable(uint32_t binding) { return m_allocatableRanges.find(binding) != m_allocatableRanges.end(); }

	inline AddressAllocator* getBindingAllocator(uint32_t binding) 
	{ 
		auto range = m_allocatableRanges.find(binding);
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return nullptr;
		return range->second.addressAllocator.get(); 
	}

	// main methods

#ifdef _NBL_DEBUG
	inline std::unique_lock<std::recursive_mutex> stAccessVerifyDebugGuard()
	{
		std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
		assert(tLock.owns_lock());
		return tLock;
	}
#else
	inline bool stAccessVerifyDebugGuard() { return false; }
#endif

	inline video::IGPUDescriptorSet* getDescriptorSet() { return m_descriptorSet.get(); }

	//! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
	inline size_type try_multi_allocate(const uint32_t binding, const size_type count, value_type* outAddresses) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		// we assume you've validated that the binding is allocatable before trying this
		auto allocator = getBindingAllocator(binding);

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
		}

		return unallocatedSize;
	}

	template<class Clock=typename std::chrono::steady_clock>
	inline size_type multi_allocate(const std::chrono::time_point<Clock>& maxWaitPoint, const uint32_t binding, const size_type count, value_type* outAddresses) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto range = m_allocatableRanges.find(binding);
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return count;

		// try allocate once
		size_type unallocatedSize = try_multi_allocate(binding,count,outAddresses);
		if (!unallocatedSize)
			return 0u;

		// then try to wait at least once and allocate
		auto& eventHandler = range->second.eventHandler;
		core::vector<IGPUDescriptorSet::SDropDescriptorSet> nulls;
		do
		{
			// FUTURE TODO: later we could only nullify the descriptors we don't end up reallocating if without robustness features
			nulls.resize(m_totalDeferredFrees);
			auto outNulls = nulls.data();
			eventHandler->wait(maxWaitPoint, unallocatedSize, outNulls);
			m_logicalDevice->nullifyDescriptors({ nulls.data(),outNulls });

			// always call with the same parameters, otherwise this turns into a mess with the non invalid_address gaps
			unallocatedSize = try_multi_allocate(binding,count,outAddresses);
			if (!unallocatedSize)
				break;
		} while(Clock::now()<maxWaitPoint);

		return unallocatedSize;
	}

	// default timeout overload
	inline size_type multi_allocate(const uint32_t binding, const size_type count, value_type* outAddresses) noexcept
	{
		// check that the binding is allocatable is done inside anyway
		return multi_allocate(TimelineEventHandlerBase::default_wait(), binding, count, outAddresses);
	}

	// Very explicit low level call you'd need to sync and drop descriptors by yourself
	// Returns: the one-past the last `outNullify` write pointer, this allows you to work out how many descriptors were freed
	inline IGPUDescriptorSet::SDropDescriptorSet* multi_deallocate(IGPUDescriptorSet::SDropDescriptorSet* outNullify, uint32_t binding, size_type count, const size_type* addr)
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		auto allocator = getBindingAllocator(binding);
		if (allocator)
		{
			for (size_type i = 0; i < count; i++)
			{
				if (addr[i] == AddressAllocator::invalid_address)
					continue;

				allocator->free_addr(addr[i], 1);
				outNullify->dstSet = m_descriptorSet.get();
				outNullify->binding = binding;
				outNullify->arrayElement = i;
				outNullify->count = 1;
				outNullify++;
			}
		}
		return outNullify;
	}

	// 100% will defer
	inline void multi_deallocate(uint32_t binding, const ISemaphore::SWaitInfo& futureWait, DeferredFreeFunctor&& functor) noexcept
	{
		auto range = m_allocatableRanges.find(binding);
		// Check if this binding has an allocator
		if (range == m_allocatableRanges.end())
			return;

		auto& eventHandler = range->second.eventHandler;
		auto debugGuard = stAccessVerifyDebugGuard();
		m_totalDeferredFrees += functor.getWorstCaseCount();
		eventHandler->latch(futureWait,std::move(functor));
	}

	// defers based on the conservative estimation if `futureWait` needs to be waited on, if doesn't will call nullify descriiptors internally immediately
	inline void multi_deallocate(uint32_t binding, size_type count, const value_type* addr, const ISemaphore::SWaitInfo& futureWait) noexcept
	{
		if (futureWait.semaphore)
			multi_deallocate(binding, futureWait, DeferredFreeFunctor(this, binding, count, addr));
		else
		{
			core::vector<IGPUDescriptorSet::SDropDescriptorSet> nulls(count);
			auto actualEnd = multi_deallocate(nulls.data(), binding, count, addr);
			// This is checked to be valid above
			auto range = m_allocatableRanges.find(binding);
			m_logicalDevice->nullifyDescriptors({nulls.data(),actualEnd});
		}
	}

	//! Returns free events still outstanding
	inline uint32_t cull_frees() noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();
		uint32_t frees = 0;
		core::vector<IGPUDescriptorSet::SDropDescriptorSet> nulls(m_totalDeferredFrees);
		auto outNulls = nulls.data();
		for (uint32_t i = 0; i < m_allocatableRanges.size(); i++)
		{
			auto& it = m_allocatableRanges[i];
			frees += it.eventHandler->poll(outNulls).eventsLeft;
		}
		m_logicalDevice->nullifyDescriptors({nulls.data(),outNulls});
		return frees;
	}
};

}

#endif
