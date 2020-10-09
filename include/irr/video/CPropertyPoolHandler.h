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
        _IRR_STATIC_INLINE_CONSTEXPR auto MaxPropertiesPerCS = 15; // TODO: Remove this and make flexible

        //
		inline uint32_t getPipelineCount() const { return m_pipelineCount; }
        //
		inline IGPUComputePipeline* getPipeline(uint32_t ix) { return m_pipelines[ix].get(); }
		inline const IGPUComputePipeline* getPipeline(uint32_t ix) const { return m_pipelines[ix].get(); }


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
				const IPropertyPool* const* pool = poolsBegin;
				uint32_t localPropID = 0u;
				auto copyPass = [&](uint32_t propertiesThisPass) -> void
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

					auto pipeline = m_pipelines[propertiesThisPass-1].get();
					m_driver->bindComputePipeline(pipeline);

					// update desc sets
					IGPUDescriptorSet* sets[2] = { m_elementDS.get(),nullptr };
					{
						if (sets[1])
						{
							m_elementDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_descriptorSetLayout));
							IGPUDescriptorSet::SDescriptorInfo info;
							info.desc = core::smart_refctd_ptr<asset::IDescriptor>(m_driver->getDefaultUpStreamingBuffer()->getBuffer());
							info.buffer = {0u,69u};
							IGPUDescriptorSet::SWriteDescriptorSet dsWrite;
							dsWrite.dstSet = m_elementDS.get();
							dsWrite.binding = 0u;
							dsWrite.arrayElement = 0u;
							dsWrite.count = 1u;
							dsWrite.descriptorType = asset::EDT_STORAGE_BUFFER;
							dsWrite.info = &info;
							m_driver->updateDescriptorSets(1u,&dsWrite,0u,nullptr);
						}
						else
						{
							success = false;
							return;
						}
					}

					// bind desc sets
					m_driver->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,2u,sets,nullptr);
		
					// dispatch
					m_driver->dispatch((maxElements+IdealWorkGroupSize-1u)/IdealWorkGroupSize,propertiesThisPass,1u);
				};

				const auto fullPasses = totalProps/MaxPropertiesPerCS;
				for (uint32_t i=0; i<fullPasses; i++)
				{
					copyPass(MaxPropertiesPerCS);
				}

				const auto leftOverProps = totalProps-fullPasses*MaxPropertiesPerCS;
				if (leftOverProps)
					copyPass(leftOverProps);
			}

			return success;
		}


		_IRR_STATIC_INLINE_CONSTEXPR auto IdealWorkGroupSize = 256u;

        IVideoDriver* m_driver;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_descriptorSetLayout;
		// TODO: Cycle through Descriptor Sets so we dont overstep
		struct
		{
			//FAT;
		};
		core::smart_refctd_ptr<IGPUDescriptorSet> m_copyBuffersDS[MaxPropertiesPerCS];
		//
        core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[MaxPropertiesPerCS];
		uint32_t m_pipelineCount;
};


}
}

#endif