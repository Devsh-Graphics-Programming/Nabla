#ifndef _NBL_ASSET_I_MESH_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_MESH_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/RasterizationStates.h"
#include "nbl/asset/IPipeline.h"

#include "nbl/asset/IRasterizationPipeline.h"


namespace nbl::asset {
    class IMeshPipelineBase : public virtual core::IReferenceCounted {
    public:
        constexpr static inline uint8_t MESH_SHADER_STAGE_COUNT = 3u;
        struct SCachedCreationParams final : public IRasterizationPipelineBase::SCachedCreationParams
        {
            uint8_t requireFullSubgroups = false;
        };
    };

    template<typename PipelineLayoutType, typename RenderpassType>
    class IMeshPipeline : public IRasterizationPipeline<PipelineLayoutType, RenderpassType>, public IMeshPipelineBase {
    protected:
        using renderpass_t = RenderpassType;
    public:
        inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }

        static inline bool hasRequiredStages(const core::bitflag<hlsl::ShaderStage>& stagePresence)
        {
            return stagePresence.hasFlags(hlsl::ShaderStage::ESS_MESH);
        }
        
    protected:
        explicit IMeshPipeline(PipelineLayoutType* layout, const IMeshPipelineBase::SCachedCreationParams& cachedParams, renderpass_t* renderpass) :
            IRasterizationPipeline<PipelineLayoutType, renderpass_t>(layout, renderpass),
            m_params(cachedParams)
        {}

        SCachedCreationParams m_params = {};
    };

}


#endif