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
		inline IGPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t ix) { return m_perPropertyCountItems[ix].descriptorSetLayout.get(); }
		inline const IGPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t ix) const { return m_perPropertyCountItems[ix].descriptorSetLayout.get(); }


		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		bool addProperties(IPropertyPool* const* poolsBegin, IPropertyPool* const* poolsEnd, uint32_t* const* indicesBegin, uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd);

        //
		inline bool uploadProperties(const IPropertyPool* const* poolsBegin, const IPropertyPool* const* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd)
		{
			return transferProperties(false,poolsBegin,poolsEnd,indicesBegin,indicesEnd,dataBegin,dataEnd);
		}

        //
		bool downloadProperties(const IPropertyPool* const* poolsBegin, const IPropertyPool* const* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, void* const* const* dataBegin, void* const* const* dataEnd)
		{
			return transferProperties(true,poolsBegin,poolsEnd,indicesBegin,indicesEnd,dataBegin,dataEnd);
		}

    protected:
		~CPropertyPoolHandler()
		{
			// pipelines drop themselves automatically
		}

		template<typename T>
		inline bool transferProperties(bool download, const IPropertyPool* const* poolsBegin, const IPropertyPool* const* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, T dataBegin, T dataEnd)
		{
			uint32_t totalProps = 0u;
			for (auto it=poolsBegin; it!=poolsEnd; it++)
				totalProps += (*it)->getPropertyCount();

			bool success = true;
			if (totalProps!=0u)
			{
				const uint32_t maxPropertiesPerPass = m_perPropertyCountItems.size();
				const auto fullPasses = totalProps/maxPropertiesPerPass;

				auto upBuff = m_driver->getDefaultUpStreamingBuffer();
				auto downBuff = m_driver->getDefaultDownStreamingBuffer();

				const IPropertyPool* const* pool = poolsBegin;
				uint32_t localPropID = 0u;

				//
				auto copyPass = [&](uint32_t propertiesThisPass)
				{
					uint32_t maxElements = 0u;
					// allocate indices and data
					{
						std::chrono::nanoseconds maxWait;
						uint32_t outaddresses, bytesize, alignment;
						m_driver->getDefaultUpStreamingBuffer()->multi_alloc(maxWait,2u,outaddresses,bytesize,alignment);
					}
					// upload indices (and data if !download)
					for (uint32_t i=0; i<propertiesThisPass; i++)
					{
						(*pool)->
					}
					uint elementCount[_IRR_BUILTIN_PROPERTY_COUNT_];
					int propertyDWORDsize_upDownFlag[_IRR_BUILTIN_PROPERTY_COUNT_];
					uint indexOffset[_IRR_BUILTIN_PROPERTY_COUNT_];
					uint indices[];

					const auto pipelineIndex = propertiesThisPass-1u;
					auto& items = m_perPropertyCountItems[pipelineIndex];
					auto pipeline = items.pipeline.get();
					m_driver->bindComputePipeline(pipeline);

					// update desc sets
					auto set = items.descriptorSetCache.getNextSet(
						m_driver
						offsets[0],sizes[0],
						offsets+1u,sizes+1u,
						offsets+1u+propertiesThisPass,sizes+1u+propertiesThisPass
					);
					if (!set)
					{
						success = false;
						return;
					}

					// bind desc sets
					m_driver->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&set.get(),nullptr);
		
					// dispatch (this will need to change to a cmd buffer submission with a fence)
					m_driver->dispatch((maxElements+IdealWorkGroupSize-1u)/IdealWorkGroupSize,propertiesThisPass,1u);
					auto fence = m_driver->placeFence(true);

					// deferred release resources

					items.descriptorSetCache.releaseSet(core::smart_refctd_ptr(fence),std::move(set));
					return fence;
				};

				//
				for (uint32_t i=0; i<fullPasses; i++)
				{
					copyPass(maxPropertiesPerPass);
				}

				const auto leftOverProps = totalProps-fullPasses*maxPropertiesPerPass;
				if (leftOverProps)
					copyPass(leftOverProps);
			}

			return success;
		}


		_IRR_STATIC_INLINE_CONSTEXPR auto IdealWorkGroupSize = 256u;

        IVideoDriver* m_driver;
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
					IVideoDriver* driver, uint32_t indexByteOffsets, uint32_t indexByteSizes,
					const uint32_t* uploadByteOffsets, const uint32_t* uploadByteSizes,
					const uint32_t* downloadByteOffets, const uint32_t* downloadByteSizes
				);

				void releaseSet(core::smart_refctd_ptr<IDriverFence>&& fence, core::smart_refctd_ptr<IGPUDescriptorSet>&& set);
		};
		struct PerPropertyCountItems
		{
			PerPropertyCountItems(IVideoDriver* driver, IGPUPipelineCache* pipelineCache, uint32_t propertyCount);

			DescriptorSetCache descriptorSetCache;
			core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		};
		// TODO: Optimize to only use one allocation for all these arrays
		core::vector<PerPropertyCountItems> m_perPropertyCountItems;
        core::vector<uint32_t> m_tmpSizes,m_alignments;
};


}
}

#endif