// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_DESCRIPTOR_SET_CACHE_H_INCLUDED_
#define _NBL_VIDEO_I_DESCRIPTOR_SET_CACHE_H_INCLUDED_


#include "nbl/asset/asset.h"

#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IDescriptorPool.h"


namespace nbl::video
{

class IDescriptorSetCache : public core::IReferenceCounted
{
	public:
		using DescSetAllocator = core::PoolAddressAllocatorST<uint32_t>;

		//
		static inline core::smart_refctd_ptr<IDescriptorSetCache> create(
			const uint32_t capacity, const IDescriptorPool::E_CREATE_FLAGS flags,
			core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& canonicalLayout
		)
		{
			if (capacity==0 || !canonicalLayout)
				return nullptr;
			void* reserved = malloc(DescSetAllocator::reserved_size(1u,capacity,1u));
			if (!reserved)
				return nullptr;
			auto* cache = new core::smart_refctd_ptr<IGPUDescriptorSet>[capacity];
			if (!cache)
				return nullptr;
			auto device = const_cast<ILogicalDevice*>(canonicalLayout->getOriginDevice());
			if (!device)
				return nullptr;
			auto pool = device->createDescriptorPoolForDSLayouts(flags,{&canonicalLayout.get(),1},&capacity);
			if (!pool)
				return nullptr;
			return core::smart_refctd_ptr<IDescriptorSetCache>(new IDescriptorSetCache(std::move(pool),std::move(canonicalLayout),cache,reserved),core::dont_grab);
		}

		//
		inline uint32_t getCapacity() const {return m_descPool->getCapacity();}

		//
		inline const IGPUDescriptorSetLayout* getCanonicalLayout() const {return m_canonicalLayout.get();}

		//
		constexpr static inline auto invalid_index = DescSetAllocator::invalid_address;
		IGPUDescriptorSet* getSet(uint32_t setIx)
		{
			if (setIx<m_descPool->getCapacity())
				return m_cache[setIx].get();
			return nullptr;
		}

		//
		inline uint32_t acquireSet()
		{
			m_deferredReclaims.poll(DeferredDescriptorSetReclaimer::single_poll);
			return m_setAllocator.alloc_addr(1u,1u);
		}

		//
		inline void releaseSet(const ISemaphore::SWaitInfo& futureWait, const uint32_t setIx)
		{
			if (setIx==invalid_index)
				return;

			m_deferredReclaims.latch(futureWait,DeferredDescriptorSetReclaimer(this,setIx));
		}

		// only public because MultiTimelineEventHandlerST needs to know about it
		class DeferredDescriptorSetReclaimer
		{
				IDescriptorSetCache* m_cache;
				uint32_t m_setIx;

			public:
				inline DeferredDescriptorSetReclaimer(IDescriptorSetCache* _cache, const uint32_t _setIx) : m_cache(_cache), m_setIx(_setIx)
				{
				}
				DeferredDescriptorSetReclaimer(const DeferredDescriptorSetReclaimer& other) = delete;
				inline DeferredDescriptorSetReclaimer(DeferredDescriptorSetReclaimer&& other) : m_cache(nullptr), m_setIx(DescSetAllocator::invalid_address)
				{
					this->operator=(std::forward<DeferredDescriptorSetReclaimer>(other));
				}

				inline ~DeferredDescriptorSetReclaimer()
				{
				}

				DeferredDescriptorSetReclaimer& operator=(const DeferredDescriptorSetReclaimer& other) = delete;
				inline DeferredDescriptorSetReclaimer& operator=(DeferredDescriptorSetReclaimer&& other)
				{
					m_cache = other.m_cache;
					m_setIx = other.m_setIx;
					other.m_cache = nullptr;
					other.m_setIx = IDescriptorSetCache::invalid_index;
					return *this;
				}

				struct single_poll_t {};
				static inline single_poll_t single_poll;
				inline bool operator()(single_poll_t _single_poll)
				{
					operator()();
					return true;
				}

				struct exhaustive_poll_t {};
				static inline exhaustive_poll_t exhaustive_poll;
				inline bool operator()(exhaustive_poll_t _exhaustive_poll)
				{
					operator()();
					return false;
				}

				inline void operator()()
				{
					#ifdef _NBL_DEBUG
					assert(m_cache && m_setIx<m_cache->getCapacity());
					#endif // _NBL_DEBUG
					m_cache->m_setAllocator.free_addr(m_setIx,1u);
				}
		};

	protected:
		friend class DeferredDescriptorSetReclaimer;
		inline IDescriptorSetCache(
			core::smart_refctd_ptr<IDescriptorPool>&& pool,
			core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& canonicalLayout,
			core::smart_refctd_ptr<IGPUDescriptorSet>* cache,
			void* const reserved
		) : m_descPool(std::move(pool)), m_canonicalLayout(std::move(canonicalLayout)), m_cache(cache),
			m_reserved(reserved), m_setAllocator(m_reserved,0u,0u,1u,m_descPool->getCapacity(),1u),
			m_deferredReclaims(const_cast<ILogicalDevice*>(m_descPool->getOriginDevice()))
		{}
		virtual inline ~IDescriptorSetCache()
		{
			// normally the dtor would do this, but we need all the events to run before we delete the storage they reference
			while (m_deferredReclaims.wait(std::chrono::steady_clock::now()+std::chrono::microseconds(100))) {}
			free(m_reserved);
			delete[] m_cache;
		}

		core::smart_refctd_ptr<IDescriptorPool> m_descPool;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_canonicalLayout;
		core::smart_refctd_ptr<IGPUDescriptorSet>* m_cache;
		void* m_reserved;
		DescSetAllocator m_setAllocator;
		MultiTimelineEventHandlerST<DeferredDescriptorSetReclaimer> m_deferredReclaims;
};

}

#endif