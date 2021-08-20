// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__


#include "nbl/asset/asset.h"

#include "nbl/video/IDescriptorPool.h"


namespace nbl::video
{

	
// TODO: factor this out
class DescriptorSetCache : public core::IReferenceCounted
{
	public:
		inline DescriptorSetCache(core::smart_refctd_ptr<IDescriptorPool>&& _descPool, core::smart_refctd_ptr<IGPUDescriptorSet>&& _canonicalDS)
			:	m_descPool(std::move(_descPool)), m_canonicalDS(std::move(_canonicalDS)), m_reserved(malloc(DescSetAllocator::reserved_size(1u,m_descPool->getCapacity(),1u))),
				m_setAllocator(m_reserved,0u,0u,1u,m_descPool->getCapacity(),1u), m_deferredReclaims()
		{
			m_cache = new core::smart_refctd_ptr<IGPUDescriptorSet>[m_descPool->getCapacity()];
		}

		//
		//inline IGPUDescriptorSet* getCanonicalDescriptorSet() {return m_canonicalDS.get();}
		inline const IGPUDescriptorSet* getCanonicalDescriptorSet() const {return m_canonicalDS.get();}

		//
		DescriptorSetCache(const DescriptorSetCache&) = delete;
		inline DescriptorSetCache(DescriptorSetCache&& other) : DescriptorSetCache()
		{
			operator=(std::move(other));
		}

		DescriptorSetCache& operator=(const DescriptorSetCache& other) = delete;
		inline DescriptorSetCache& operator=(DescriptorSetCache&& other)
		{
			std::swap(m_descPool,other.m_descPool);
			std::swap(m_canonicalDS,other.m_canonicalDS);
			std::swap(m_reserved,other.m_reserved);
			std::swap(m_setAllocator,other.m_setAllocator);
			std::swap(m_cache,other.m_cache);
			return *this;
		}

		//
		IGPUDescriptorSet* getSet(uint32_t setIx)
		{
			if (setIx<m_descPool->getCapacity())
				return m_cache[setIx].get();
			return nullptr;
		}

		//
		inline uint32_t acquireSet()
		{
			return m_setAllocator.alloc_addr(1u,1u);
		}

		//
		void releaseSet(core::smart_refctd_ptr<IGPUFence>&& fence, uint32_t setIx)
		{
		}

		// only public because GPUDeferredEventHandlerST needs to know about it
		class DeferredDescriptorSetReclaimer
		{
				core::vector<core::smart_refctd_ptr<IGPUDescriptorSet>>* unusedSets;
				core::smart_refctd_ptr<IGPUDescriptorSet> set;

			public:
				inline DeferredDescriptorSetReclaimer(core::vector<core::smart_refctd_ptr<IGPUDescriptorSet>>* _unusedSets, core::smart_refctd_ptr<IGPUDescriptorSet>&& _set)
														: unusedSets(_unusedSets), set(std::move(_set))
				{
				}
				DeferredDescriptorSetReclaimer(const DeferredDescriptorSetReclaimer& other) = delete;
				DeferredDescriptorSetReclaimer(DeferredDescriptorSetReclaimer&& other) : unusedSets(nullptr), set()
				{
					this->operator=(std::forward<DeferredDescriptorSetReclaimer>(other));
				}

				inline ~DeferredDescriptorSetReclaimer()
				{
				}

				DeferredDescriptorSetReclaimer& operator=(const DeferredDescriptorSetReclaimer& other) = delete;
				inline DeferredDescriptorSetReclaimer& operator=(DeferredDescriptorSetReclaimer&& other)
				{
					unusedSets = other.unusedSets;
					set   = std::move(other.set);
					other.unusedSets = nullptr;
					other.set   = nullptr;
					return *this;
				}

				struct single_poll_t {};
				static inline single_poll_t single_poll;
				inline bool operator()(single_poll_t _single_poll)
				{
					operator()();
					return true;
				}

				inline void operator()()
				{
					#ifdef _NBL_DEBUG
					assert(unusedSets && set.get());
					#endif // _NBL_DEBUG
					unusedSets->push_back(std::move(set));
				}
		};
	protected:
		inline DescriptorSetCache() : m_descPool(nullptr), m_canonicalDS(nullptr), m_reserved(nullptr), m_setAllocator(), m_cache(nullptr), m_deferredReclaims() {}
		~DescriptorSetCache()
		{
			// destructor of `deferredReclaims` will wait for all fences
			delete[] m_cache;
			free(m_reserved);
		}

		core::smart_refctd_ptr<IDescriptorPool> m_descPool;
		core::smart_refctd_ptr<IGPUDescriptorSet> m_canonicalDS;
		void* m_reserved;
		using DescSetAllocator = core::PoolAddressAllocatorST<uint32_t>;
		DescSetAllocator m_setAllocator;
		core::smart_refctd_ptr<IGPUDescriptorSet>* m_cache;
		GPUDeferredEventHandlerST<DeferredDescriptorSetReclaimer> m_deferredReclaims;
};



class IPropertyPool;

// property pool factory is externally synchronized
class CPropertyPoolHandler final : public core::IReferenceCounted, public core::Unmovable
{
	public:
		//
		CPropertyPoolHandler(core::smart_refctd_ptr<ILogicalDevice>&& device);

        _NBL_STATIC_INLINE_CONSTEXPR auto MinimumPropertyAlignment = alignof(uint32_t);

        //
		inline IGPUComputePipeline* getPipeline() {return m_pipeline.get();}
		inline const IGPUComputePipeline* getPipeline() const {return m_pipeline.get();}
        //
		//inline IGPUDescriptorSet* getCanonicalDescriptorSet() { return m_dsCache.getCanonicalDescriptorSet(); }
		inline const IGPUDescriptorSet* getCanonicalDescriptorSet() const { return m_dsCache.getCanonicalDescriptorSet(); }

		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		struct AllocationRequest
		{
			AllocationRequest() : pool(nullptr), outIndices{nullptr,nullptr}, data(nullptr) {}
			AllocationRequest(IPropertyPool* _pool, core::SRange<uint32_t> _outIndices, const void* const* _data) : pool(_pool), outIndices(_outIndices), data(_data) {}

			IPropertyPool* pool;
			core::SRange<uint32_t> outIndices;
			const void* const* data; 
		};
		// returns false if an allocation or part of a transfer has failed
		// while its possible to detect which allocation has failer, its not possible to know exactly what transfer failed
		bool addProperties(IGPUCommandBuffer* cmdbuf, const AllocationRequest* requestsBegin, const AllocationRequest* requestsEnd);

        //
		struct TransferRequest
		{
			TransferRequest() : download(false), pool(nullptr), indices{nullptr,nullptr}, propertyID(0xdeadbeefu)
			{
				readData = nullptr;
			}

			bool download;
			IPropertyPool* pool;
			core::SRange<const uint32_t> indices;
			uint32_t propertyID;
			union
			{
				const void* readData;
				void* writeData;
			};
		};
		bool transferProperties(IGPUCommandBuffer* cmdbuf, const TransferRequest* requestsBegin, const TransferRequest* requestsEnd);

    protected:
		~CPropertyPoolHandler()
		{
			free(m_tmpIndexRanges);
			// pipelines drop themselves automatically
		}

		static inline constexpr auto IdealWorkGroupSize = 256u;
		static inline constexpr auto DescriptorCacheSize = 128u;


		core::smart_refctd_ptr<ILogicalDevice> m_device;
		struct IndexUploadRange
		{
			IndexUploadRange() : source{nullptr,nullptr}, destOff(0xdeadbeefu) {}

			core::SRange<const uint32_t> source;
			uint32_t destOff;
		};
        IndexUploadRange* m_tmpIndexRanges;
		uint32_t* m_tmpAddresses,* m_tmpSizes,* m_alignments;
		uint8_t m_maxPropertiesPerPass;

		// TODO: investigate using Push Descriptors for this
		class TransferDescriptorSetCache : public DescriptorSetCache
		{
			public:
				TransferDescriptorSetCache(ILogicalDevice* const device, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout, uint32_t maxPropertiesPerPass);
				//
				IGPUDescriptorSet* getNextSet(const TransferRequest* requests, uint32_t parameterBufferSize, const uint32_t* uploadAddresses, const uint32_t* downloadAddresses);
		};
		TransferDescriptorSetCache m_dsCache;

		core::smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
};


}

#endif