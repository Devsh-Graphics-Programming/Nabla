#ifndef _NBL_ASSET_I_MESH_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_MESH_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/RasterizationStates.h"
#include "nbl/asset/IPipeline.h"


namespace nbl::asset {
    class IMeshPipelineBase : public virtual core::IReferenceCounted {
    public:
        constexpr static inline uint8_t MESH_SHADER_STAGE_COUNT = 3u;
        struct SCachedCreationParams final {
            SRasterizationParams rasterization = {};
            SBlendParams blend = {};
            uint32_t subpassIx = 0u;
            uint8_t requireFullSubgroups = false;
        };

    };

    template<typename PipelineLayoutType, typename RenderpassType>
    class IMeshPipeline : public IPipeline<PipelineLayoutType>, public IMeshPipelineBase {
    protected:
        using renderpass_t = RenderpassType;
    public:
        inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }
        inline const renderpass_t* getRenderpass() const {return m_renderpass.get();}

        static inline bool hasRequiredStages(const core::bitflag<hlsl::ShaderStage>& stagePresence)
        {
            return stagePresence.hasFlags(hlsl::ShaderStage::ESS_MESH);
        }
        
    protected:
        explicit IMeshPipeline(PipelineLayoutType* layout, const SCachedCreationParams& cachedParams, renderpass_t* renderpass) :
            IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<PipelineLayoutType>(layout)),
            m_params(cachedParams), m_renderpass(core::smart_refctd_ptr<renderpass_t>(renderpass))
        {
        }

        SCachedCreationParams m_params = {};
        core::smart_refctd_ptr<renderpass_t> m_renderpass = nullptr;
    };

}


#endif