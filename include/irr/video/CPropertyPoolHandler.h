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

		
		using transfer_result_t = std::pair<bool, core::smart_refctd_ptr<IDriverFence> >;
		#define DEFAULT_WAIT std::chrono::nanoseconds(50000ull)

		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		struct AllocationRequest
		{
			private:
				bool reserved = false;
			public:
				uint32_t propertyID;
				IPropertyPool* pool;
				core::SRange<uint32_t> outIndices;
				const void* data; 
		};
		transfer_result_t addProperties(const AllocationRequest* requestsBegin, const AllocationRequest* requestsEnd, const std::chrono::nanoseconds& maxWait=DEFAULT_WAIT);

        //
		struct TransferRequest
		{
			public:
				bool download = false;
				uint32_t propertyID;
				IPropertyPool* pool;
				core::SRange<const uint32_t> indices;
				union
				{
					const void* readData;
					void* writeData;
				};
		};
		transfer_result_t transferProperties(const TransferRequest* requestsBegin, const TransferRequest* requestsEnd, const std::chrono::nanoseconds& maxWait=DEFAULT_WAIT);
		#undef DEFAULT_WAIT

    protected:
		~CPropertyPoolHandler()
		{
			// pipelines drop themselves automatically
		}


		_IRR_STATIC_INLINE_CONSTEXPR auto IdealWorkGroupSize = 256u;


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
					IVideoDriver* driver, const TransferRequest* requests, uint32_t parameterBufferSize, const uint32_t* uploadAddresses, const uint32_t* downloadAddresses
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
		struct IndexUploadRange
		{
			core::SRange<const uint32_t> source;
			uint32_t destOff;
		};
        core::vector<IndexUploadRange> m_tmpIndexRanges;
        core::vector<uint32_t> m_tmpAddresses,m_tmpSizes,m_alignments;
};


}
}

#endif