// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_DESCRIPTOR_SET_CACHE_H_INCLUDED__
#define __NBL_VIDEO_I_DESCRIPTOR_SET_CACHE_H_INCLUDED__


#include "nbl/asset/asset.h"

#include "nbl/video/IGPUFence.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IDescriptorPool.h"


namespace nbl::video
{

	
class IDescriptorSetCache : public core::IReferenceCounted
{
	public:
		using DescSetAllocator = core::PoolAddressAllocatorST<uint32_t>;

		IDescriptorSetCache(ILogicalDevice* device, core::smart_refctd_ptr<IDescriptorPool>&& _descPool, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _canonicalLayout);

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
			m_deferredReclaims.pollForReadyEvents(DeferredDescriptorSetReclaimer::single_poll);
			return m_setAllocator.alloc_addr(1u,1u);
		}

		// needs to be called before you reset any fences which latch the deferred release
		inline void poll_all()
		{
			m_deferredReclaims.pollForReadyEvents(DeferredDescriptorSetReclaimer::exhaustive_poll);
		}

		//
		inline void releaseSet(ILogicalDevice* device, core::smart_refctd_ptr<IGPUFence>&& fence, const uint32_t setIx)
		{
			if (setIx==invalid_index)
				return;

			m_deferredReclaims.addEvent(GPUEventWrapper(device,std::move(fence)),DeferredDescriptorSetReclaimer(this,setIx));
		}

		// only public because GPUDeferredEventHandlerST needs to know about it
		class DeferredDescriptorSetReclaimer
		{
				IDescriptorSetCache* m_cache;
				uint32_t m_setIx;

			public:
				inline DeferredDescriptorSetReclaimer(IDescriptorSetCache* _cache, const uint32_t _setIx) : m_cache(_cache), m_setIx(_setIx)
				{
				}
				DeferredDescriptorSetReclaimer(const DeferredDescriptorSetReclaimer& other) = delete;
				DeferredDescriptorSetReclaimer(DeferredDescriptorSetReclaimer&& other) : m_cache(nullptr), m_setIx(DescSetAllocator::invalid_address)
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
		IDescriptorSetCache(ILogicalDevice* device, const uint32_t capacity);
		~IDescriptorSetCache()
		{
			// destructor of `deferredReclaims` will wait for all fences
			free(m_reserved);
			delete[] m_cache;
		}

		core::smart_refctd_ptr<IDescriptorPool> m_descPool;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_canonicalLayout;
		core::smart_refctd_ptr<IGPUDescriptorSet>* m_cache;
		void* m_reserved;
		DescSetAllocator m_setAllocator;
		GPUDeferredEventHandlerST<DeferredDescriptorSetReclaimer> m_deferredReclaims;
};

}

#endif