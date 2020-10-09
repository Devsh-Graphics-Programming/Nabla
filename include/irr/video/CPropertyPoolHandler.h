#ifndef __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__


#include "irr/asset/asset.h"

#include "IVideoDriver.h"
#include "irr/video/IGPUComputePipeline.h"


namespace irr
{
namespace video
{

class IPropertyPool;

// property pool factory is externally synchronized
class CPropertyPoolHandler final : public core::IReferenceCounted
{
	public:
		CPropertyPoolHandler(IVideoDriver* driver, IGPUPipelineCache* pipelineCache);

        _IRR_STATIC_INLINE_CONSTEXPR auto MinimumPropertyAlignment = alignof(uint32_t);

        //
		inline uint32_t getPipelineCount() const { return m_perPropertyCountItems.size(); }
        //
		inline IGPUComputePipeline* getPipeline(uint32_t ix) { return m_perPropertyCountItems[ix].pipeline.get(); }
		inline const IGPUComputePipeline* getPipeline(uint32_t ix) const { return m_perPropertyCountItems[ix].pipeline.get(); }
        //
		inline IGPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t ix) { return m_perPropertyCountItems[ix].descriptorSetCache.getLayout().get(); }
		inline const IGPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t ix) const { return m_perPropertyCountItems[ix].descriptorSetCache.getLayout().get(); }


		using transfer_result_t = std::pair<bool,core::smart_refctd_ptr<IDriverFence> >;
		#define DEFAULT_WAIT std::chrono::nanoseconds(50000ull)
		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		transfer_result_t addProperties(IPropertyPool* const* poolsBegin, IPropertyPool* const* poolsEnd, uint32_t* const* indicesBegin, uint32_t* const* indicesEnd, const void* const* const* data, const std::chrono::nanoseconds& maxWait=DEFAULT_WAIT);

        //
		inline transfer_result_t uploadProperties(const IPropertyPool* const* poolsBegin, const IPropertyPool* const* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, const void* const* const* data, const std::chrono::nanoseconds& maxWait=DEFAULT_WAIT)
		{
			return transferProperties(false,poolsBegin,poolsEnd,indicesBegin,indicesEnd,data,maxWait);
		}

        //
		inline transfer_result_t downloadProperties(const IPropertyPool* const* poolsBegin, const IPropertyPool* const* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, void* const* const* data, const std::chrono::nanoseconds& maxWait=DEFAULT_WAIT)
		{
			return transferProperties(true,poolsBegin,poolsEnd,indicesBegin,indicesEnd,data,maxWait);
		}
		#undef DEFAULT_WAIT

    protected:
		~CPropertyPoolHandler()
		{
			// pipelines drop themselves automatically
		}

		template<typename T>
		inline transfer_result_t transferProperties(bool download, const IPropertyPool* const* poolsBegin, const IPropertyPool* const* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, T* const* const* data, const std::chrono::nanoseconds& maxWait)
		{
			const auto poolCount = std::distance(poolsBegin,poolsEnd);

			uint32_t totalProps = 0u;
			for (auto i=0u; i<poolCount; i++)
				totalProps += poolsBegin[i]->getPropertyCount();

			transfer_result_t retval = { true,nullptr };
			if (totalProps!=0u)
			{
				const uint32_t maxPropertiesPerPass = m_perPropertyCountItems.size();
				const auto fullPasses = totalProps/maxPropertiesPerPass;

				auto upBuff = m_driver->getDefaultUpStreamingBuffer();
				auto downBuff = m_driver->getDefaultDownStreamingBuffer();

				auto poolIt = poolsBegin;
				uint32_t localPropID = 0u;
				
				auto maxWaitPoint = std::chrono::high_resolution_clock::now()+maxWait; // 50 us
				//
				auto copyPass = [&](uint32_t propertiesThisPass) -> void
				{
					const uint32_t headerSize = sizeof(uint32_t)*3u*propertiesThisPass;

					constexpr auto invalid_address = std::remove_reference_t<decltype(upBuff->getAllocator())>::invalid_address;
					const uint32_t upAllocations = 1u+(download ? 0u:propertiesThisPass);

					const auto poolsLocalBegin = poolIt;
					uint32_t distinctPools = 1u;
					m_tmpSizes[0u] = sizeof(uint32_t)*3u*propertiesThisPass; // TODO
					for (uint32_t i=0; i<propertiesThisPass; i++)
					{
						const IPropertyPool* pool = *poolIt;
						const auto poolID = std::distance(poolsBegin,poolIt);

						m_transientPassData[i] = {&pool->getMemoryBlock(),data[poolID][localPropID],pool->getPropertySize(localPropID)};

						const auto elements = std::distance(indicesBegin[poolID],indicesEnd[poolID]);
						assert(elements);
						m_tmpSizes[i+1u] = elements*m_transientPassData[i].propSize;

						if ((++localPropID) >= pool->getPropertyCount())
						{
							localPropID = 0u;
							poolIt++;
							assert(poolIt!=poolsEnd);
							distinctPools++;
						}
					}

					// allocate indices and upload/allocate data
					uint32_t maxElements = 0u;
					{
						std::fill(m_tmpAddresses.begin(),m_tmpAddresses.begin()+upAllocations,invalid_address);
						upBuff->multi_alloc(maxWaitPoint,upAllocations,m_tmpAddresses.data(),m_tmpSizes.data(),m_alignments.data());

						if (download)
							downBuff->multi_alloc(maxWaitPoint,propertiesThisPass,m_tmpAddresses.data()+1u,m_tmpSizes.data()+1u,m_alignments.data());
						
						// upload
						for (uint32_t i=1u; i<=upAllocations; i++)
						if (m_tmpAddresses[i]!=invalid_address)
							memcpy(reinterpret_cast<uint8_t*>(upBuff->getBufferPointer())+m_tmpAddresses[i],m_transientPassData[i].data,m_tmpSizes[i]);
						
						auto* indexBufferPtr = reinterpret_cast<uint32_t*>(upBuff->getBufferPointer())+m_tmpAddresses[0u]/sizeof(uint32_t);
						// write `elementCount`
						for (uint32_t i=0; i<propertiesThisPass; i++)
							*(indexBufferPtr++) = m_tmpSizes[i+1u]/m_transientPassData[i].propSize;
						// write `propertyDWORDsize_upDownFlag`
						for (uint32_t i=0; i<propertiesThisPass; i++)
							*reinterpret_cast<int32_t*>(indexBufferPtr++) = (download ? -1:1)*m_transientPassData[i].propSize;
						// write `indexOffset`
						for (uint32_t i=0; i<propertiesThisPass; i++)
							*(indexBufferPtr++) = m_transientPassData[i].indexOffset;
						// write the indices
						for (uint32_t i=0; i<distinctPools; i++)
						{
							const auto poolID = std::distance(poolsBegin,poolsLocalBegin)+i;
							const auto indexCount = indicesEnd[poolID]-indicesBegin[poolID];
							maxElements = core::max(indexCount,maxElements);
							memcpy(indexBufferPtr,indicesBegin[poolID],indexCount);
							indexBufferPtr += indexCount;
						}
					}

					const auto pipelineIndex = propertiesThisPass-1u;
					auto& items = m_perPropertyCountItems[pipelineIndex];
					auto pipeline = items.pipeline.get();
					m_driver->bindComputePipeline(pipeline);

					// update desc sets
					auto set = items.descriptorSetCache.getNextSet(
						download,m_driver
						m_tmpAddresses[0],m_tmpSizes[0],
						m_tmpAddresses.data()+1u,m_tmpSizes.data()+1u,
						m_transientPassData
					);
					if (!set)
					{
						retval.first = false;
						return;
					}

					// bind desc sets
					m_driver->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&set.get(),nullptr);
		
					// dispatch (this will need to change to a cmd buffer submission with a fence)
					m_driver->dispatch((maxElements+IdealWorkGroupSize-1u)/IdealWorkGroupSize,propertiesThisPass,1u);
					auto& fence = retval.second = m_driver->placeFence(true);

					// deferred release resources
					upBuff->multi_free(upAllocations,m_tmpAddresses.data(),m_tmpSizes.data(),core::smart_refctd_ptr(fence));
					if (download)
						downBuff->multi_free(propertiesThisPass,m_tmpAddresses.data()+1u,m_tmpSizes.data()+1u,core::smart_refctd_ptr(fence));
					items.descriptorSetCache.releaseSet(core::smart_refctd_ptr(fence),std::move(set));
				};

				//
				core::smart_refctd_ptr<IDriverFence> fence;
				for (uint32_t i=0; i<fullPasses; i++)
				{
					copyPass(maxPropertiesPerPass);
				}

				const auto leftOverProps = totalProps-fullPasses*maxPropertiesPerPass;
				if (leftOverProps)
					copyPass(leftOverProps);
			}

			return retval;
		}


		_IRR_STATIC_INLINE_CONSTEXPR auto IdealWorkGroupSize = 256u;

		struct TransientPassData
		{
			const asset::SBufferRange<IGPUBuffer>* memBlock;
			const void* data;
			int32_t propSize;
			uint32_t indexOffset;
		};
		class DescriptorSetCache
		{
				class DeferredDescriptorSetReclaimer
				{
						DescriptorSetCache*   cache;
						core::smart_refctd_ptr<IGPUDescriptorSet> set;

					public:
						inline DeferredDescriptorSetReclaimer(DescriptorSetCache* _this, core::smart_refctd_ptr<IGPUDescriptorSet>&& _set)
															: cache(_this), set(std::move(_set))
						{
						}
						DeferredDescriptorSetReclaimer(const DeferredDescriptorSetReclaimer& other) = delete;
						inline DeferredDescriptorSetReclaimer(DeferredDescriptorSetReclaimer&& other) : cache(nullptr), set()
						{
							this->operator=(std::forward<DeferredDescriptorSetReclaimer>(other));
						}

						inline ~DeferredDescriptorSetReclaimer()
						{
						}

						DeferredDescriptorSetReclaimer& operator=(const DeferredDescriptorSetReclaimer& other) = delete;
						inline DeferredDescriptorSetReclaimer& operator=(DeferredDescriptorSetReclaimer&& other)
						{
							cache = other.cache;
							set   = std::move(other.set);
							other.cache = nullptr;
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
							#ifdef _IRR_DEBUG
							assert(cache && set.get());
							#endif // _IRR_DEBUG
							cache->unusedSets.push_back(std::move(set));
						}
				};
				GPUDeferredEventHandlerST<DeferredDescriptorSetReclaimer> deferredReclaims;
				core::vector<core::smart_refctd_ptr<IGPUDescriptorSet>> unusedSets;
				core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout;
				uint32_t propertyCount;
		
			public:
				DescriptorSetCache(IVideoDriver* driver, uint32_t _propertyCount);
				// ~DescriptorSetCache(); destructor of `deferredReclaims` will wait for all fences

				core::smart_refctd_ptr<IGPUDescriptorSetLayout> getLayout() const { return core::smart_refctd_ptr(layout); }

				core::smart_refctd_ptr<IGPUDescriptorSet> getNextSet(
					bool download, IVideoDriver* driver,
					uint32_t indexByteOffsets, uint32_t indexByteSizes,
					const uint32_t* cpuByteOffsets, const uint32_t* dataByteSizes,
					const TransientPassData* transientData
				);

				void releaseSet(core::smart_refctd_ptr<IDriverFence>&& fence, core::smart_refctd_ptr<IGPUDescriptorSet>&& set);
		};
		struct PerPropertyCountItems
		{
			PerPropertyCountItems(IVideoDriver* driver, IGPUPipelineCache* pipelineCache, uint32_t propertyCount);

			DescriptorSetCache descriptorSetCache;
			core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		};
		IVideoDriver* m_driver;
		// TODO: Optimize to only use one allocation for all these arrays
		core::vector<PerPropertyCountItems> m_perPropertyCountItems;
		core::vector<TransientPassData> m_transientPassData;
        core::vector<uint32_t> m_tmpAddresses,m_tmpSizes,m_alignments;
};


}
}

#endif