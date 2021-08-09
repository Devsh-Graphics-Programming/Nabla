// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__


#include "nbl/asset/asset.h"


namespace nbl::video
{
#if 0
class IPropertyPool;

// property pool factory is externally synchronized
class CPropertyPoolHandler final : public core::IReferenceCounted, public core::Unmovable
{
	public:
		CPropertyPoolHandler(ILogicalDevice* device, IGPUPipelineCache* pipelineCache);

        _NBL_STATIC_INLINE_CONSTEXPR auto MinimumPropertyAlignment = alignof(uint32_t);

        //
		inline uint32_t getPipelineCount() const { return m_perPropertyCountItems.size(); }
        //
		inline IGPUComputePipeline* getPipeline(uint32_t ix) { return m_perPropertyCountItems[ix].pipeline.get(); }
		inline const IGPUComputePipeline* getPipeline(uint32_t ix) const { return m_perPropertyCountItems[ix].pipeline.get(); }
        //
		inline IGPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t ix) { return m_perPropertyCountItems[ix].descriptorSetCache.getLayout().get(); }
		inline const IGPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t ix) const { return m_perPropertyCountItems[ix].descriptorSetCache.getLayout().get(); }

		
		using transfer_result_t = std::pair<bool, core::smart_refctd_ptr<IDriverFence> >;

		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		struct AllocationRequest
		{
			IPropertyPool* pool;
			core::SRange<uint32_t> outIndices;
			const void* const* data; 
		};
		transfer_result_t addProperties(const AllocationRequest* requestsBegin, const AllocationRequest* requestsEnd, const std::chrono::high_resolution_clock::time_point& maxWaitPoint=GPUEventWrapper::default_wait());

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
		transfer_result_t transferProperties(const TransferRequest* requestsBegin, const TransferRequest* requestsEnd, const std::chrono::high_resolution_clock::time_point& maxWaitPoint=GPUEventWrapper::default_wait());
		
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
				static single_poll_t single_poll;
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
		~CPropertyPoolHandler()
		{
			// pipelines drop themselves automatically
		}


		_NBL_STATIC_INLINE_CONSTEXPR auto IdealWorkGroupSize = 256u;


		class DescriptorSetCache
		{
				GPUDeferredEventHandlerST<DeferredDescriptorSetReclaimer> deferredReclaims;
				core::vector<core::smart_refctd_ptr<IGPUDescriptorSet>> unusedSets;
				core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout;
				uint32_t propertyCount;
		
			public:
				inline DescriptorSetCache() : deferredReclaims(), unusedSets(), layout(), propertyCount(0u) {}
				DescriptorSetCache(IVideoDriver* driver, uint32_t _propertyCount);
				DescriptorSetCache(const DescriptorSetCache&) = delete;
				inline DescriptorSetCache(DescriptorSetCache&& other) : DescriptorSetCache()
				{
					operator=(std::move(other));
				}

				// ~DescriptorSetCache(); destructor of `deferredReclaims` will wait for all fences

				DescriptorSetCache& operator=(const DescriptorSetCache& other) = delete;
				inline DescriptorSetCache& operator=(DescriptorSetCache&& other)
				{
					std::swap(deferredReclaims,other.deferredReclaims);
					std::swap(unusedSets,other.unusedSets);
					std::swap(layout,other.layout);
					std::swap(propertyCount,other.propertyCount);
					return *this;
				}


				core::smart_refctd_ptr<IGPUDescriptorSetLayout> getLayout() const { return core::smart_refctd_ptr(layout); }

				core::smart_refctd_ptr<IGPUDescriptorSet> getNextSet(
					IVideoDriver* driver, const TransferRequest* requests, uint32_t parameterBufferSize, const uint32_t* uploadAddresses, const uint32_t* downloadAddresses
				);

				void releaseSet(core::smart_refctd_ptr<IDriverFence>&& fence, core::smart_refctd_ptr<IGPUDescriptorSet>&& set);
		};
		struct PerPropertyCountItems
		{
			inline PerPropertyCountItems() : descriptorSetCache(), pipeline() {}
			PerPropertyCountItems(IVideoDriver* driver, IGPUPipelineCache* pipelineCache, uint32_t propertyCount);
			PerPropertyCountItems(const PerPropertyCountItems&) = delete;
			inline PerPropertyCountItems(PerPropertyCountItems&& other) : PerPropertyCountItems()
			{
				operator=(std::move(other));
			}

			PerPropertyCountItems& operator=(const PerPropertyCountItems&) = delete;
			inline PerPropertyCountItems& operator=(PerPropertyCountItems&& other)
			{
				std::swap(descriptorSetCache,other.descriptorSetCache);
				std::swap(pipeline,other.pipeline);
				return *this;
			}

			DescriptorSetCache descriptorSetCache;
			core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		};
		IVideoDriver* m_driver;
		// TODO: Optimize to only use one allocation for all these arrays
		core::vector<PerPropertyCountItems> m_perPropertyCountItems;
		struct IndexUploadRange
		{
			IndexUploadRange() : source{nullptr,nullptr}, destOff(0xdeadbeefu) {}

			core::SRange<const uint32_t> source;
			uint32_t destOff;
		};
        core::vector<IndexUploadRange> m_tmpIndexRanges;
        core::vector<uint32_t> m_tmpAddresses,m_tmpSizes,m_alignments;
};
#endif

}

#endif