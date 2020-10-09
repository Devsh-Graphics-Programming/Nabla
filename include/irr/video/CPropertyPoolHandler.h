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
		CPropertyPoolHandler(IVideoDriver* driver);

        _IRR_STATIC_INLINE_CONSTEXPR auto MinimumPropertyAlignment = alignof(uint32_t);
        _IRR_STATIC_INLINE_CONSTEXPR auto MaxPropertiesPerCS = 15;

        // if the pipeline for the config is in the LRU cache it will be returned, if its not then it will be created if `canCompileNew` is true
        core::smart_refctd_ptr<IGPUComputePipeline> getCopyPipeline(const PipelineKey& key, bool canCompileNew, IGPUPipelineCache* pipelineCache=nullptr);

        //
		inline uint32_t getPipelineCount() const { return m_pipelineCount; }
        //
		inline IGPUComputePipeline* getPipeline(uint32_t ix) { return m_pipelines[ix]; }
		inline const IGPUComputePipeline* getPipeline(uint32_t ix) const { return m_pipelines[ix]; }


		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		bool addProperties(const IPropertyPool* poolsBegin, const IPropertyPool* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd)

        //
		bool uploadProperties(const IPropertyPool* poolsBegin, const IPropertyPool* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, const void* const* const* dataBegin, const void* const* const* dataEnd);

        //
        bool downloadProperties(const IPropertyPool* poolsBegin, const IPropertyPool* poolsEnd, const uint32_t* const* indicesBegin, const uint32_t* const* indicesEnd, void* const* const* dataBegin, void* const* const* dataEnd)

    protected:
		~CPropertyPoolHandler()
		{
			// pipelines drop themselves automatically
		}


		_IRR_STATIC_INLINE_CONSTEXPR auto IdealWorkGroupSize = 256u;

        IVideoDriver* m_driver;
		core::smart_refctd_ptr<IGPUDescriptorSet> m_elementDS;
		core::smart_refctd_ptr<IGPUDescriptorSet> m_copyBuffersDS[MaxPropertiesPerCS];
        core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[MaxPropertiesPerCS];
		uint32_t m_pipelineCount;
};


}
}

#endif