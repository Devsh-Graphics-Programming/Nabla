// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_COMMAND_POOL_CACHE_H_INCLUDED__
#define __NBL_VIDEO_I_COMMAND_POOL_CACHE_H_INCLUDED__


#include "nbl/asset/asset.h"

#include "nbl/video/IGPUFence.h"
#include "nbl/video/IGPUCommandPool.h"


namespace nbl::video
{

	
class ICommandPoolCache : public core::IReferenceCounted
{
	public:
		using CommandPoolAllocator = core::PoolAddressAllocatorST<uint32_t>;

		ICommandPoolCache(ILogicalDevice* device, const uint32_t queueFamilyIx, const IGPUCommandPool::E_CREATE_FLAGS _flags, const uint32_t capacity);

		//
		inline uint32_t getCapacity() const {return m_cmdPoolAllocator.get_total_size();}

		//
		constexpr static inline auto invalid_index = CommandPoolAllocator::invalid_address;
		IGPUCommandPool* getPool(uint32_t poolIx)
		{
			if (poolIx<getCapacity())
				return m_cache[poolIx].get();
			return nullptr;
		}

		//
		inline uint32_t acquirePool()
		{
			m_deferredResets.pollForReadyEvents(DeferredCommandPoolResetter::single_poll);
			return m_cmdPoolAllocator.alloc_addr(1u,1u);
		}

		// needs to be called before you reset any fences which latch the deferred release
		inline void poll_all()
		{
			m_deferredResets.pollForReadyEvents(DeferredCommandPoolResetter::exhaustive_poll);
		}

		//
		inline void releaseSet(ILogicalDevice* device, core::smart_refctd_ptr<IGPUFence>&& fence, const uint32_t poolIx)
		{
			if (poolIx==invalid_index)
				return;
			
			if (fence)
				m_deferredResets.addEvent(GPUEventWrapper(device,std::move(fence)),DeferredCommandPoolResetter(this,poolIx));
			else
				releaseSet(device,poolIx);
		}

		// only public because GPUDeferredEventHandlerST needs to know about it
		class DeferredCommandPoolResetter
		{
				ICommandPoolCache* m_cache;
				uint32_t m_poolIx;

			public:
				inline DeferredCommandPoolResetter(ICommandPoolCache* _cache, const uint32_t _poolIx) : m_cache(_cache), m_poolIx(_poolIx)
				{
				}
				DeferredCommandPoolResetter(const DeferredCommandPoolResetter& other) = delete;
				DeferredCommandPoolResetter(DeferredCommandPoolResetter&& other) : m_cache(nullptr), m_poolIx(CommandPoolAllocator::invalid_address)
				{
					this->operator=(std::forward<DeferredCommandPoolResetter>(other));
				}

				inline ~DeferredCommandPoolResetter()
				{
				}

				DeferredCommandPoolResetter& operator=(const DeferredCommandPoolResetter& other) = delete;
				inline DeferredCommandPoolResetter& operator=(DeferredCommandPoolResetter&& other)
				{
					m_cache = other.m_cache;
					m_poolIx = other.m_poolIx;
					other.m_cache = nullptr;
					other.m_poolIx = ICommandPoolCache::invalid_index;
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

				void operator()();
		};

	protected:
		friend class DeferredCommandPoolResetter;
		virtual ~ICommandPoolCache()
		{
			m_deferredResets.cullEvents(0u);
			free(m_reserved);
			delete[] m_cache;
		}
		
		void releaseSet(ILogicalDevice* device, const uint32_t poolIx);

		core::smart_refctd_ptr<IGPUCommandPool>* m_cache;
		void* m_reserved;
		CommandPoolAllocator m_cmdPoolAllocator;
		GPUDeferredEventHandlerST<DeferredCommandPoolResetter> m_deferredResets;
		// TODO: after CommandPool resetting, get rid of these 
		const uint32_t m_queueFamilyIx;
		const IGPUCommandPool::E_CREATE_FLAGS m_flags;
};

}

#endif