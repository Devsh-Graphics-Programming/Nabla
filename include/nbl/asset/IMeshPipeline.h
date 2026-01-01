#ifndef _NBL_ASSET_I_MESH_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_MESH_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/RasterizationStates.h"
#include "nbl/asset/IPipeline.h"


namespace nbl::asset {
    class IMeshPipelineBase : public virtual core::IReferenceCounted {
    public:
        constexpr static inline uint8_t MESH_SHADER_STAGE_COUNT = 3u; //i dont know what this is going to be used for yet, might be redundant
        struct SCachedCreationParams final {
            SRasterizationParams rasterization = {};
            SBlendParams blend = {};
            uint32_t subpassIx = 0u; //this subpass stuff is eluding me rn. i might just need to crack open the vulkan documentation
            uint8_t requireFullSubgroups = false;
        };

    };

    template<typename PipelineLayoutType, typename RenderpassType>
    class IMeshPipeline : public IPipeline<PipelineLayoutType>, public IMeshPipelineBase {
    protected:
        using renderpass_t = RenderpassType;
        //using base_creation_params_t = IPipeline<PipelineLayoutType>;//compute uses this, idk if its necessary yet
    public:

        static inline bool hasRequiredStages(const core::bitflag<hlsl::ShaderStage>& stagePresence)
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-stage-02096
            if (!stagePresence.hasFlags(hlsl::ShaderStage::ESS_MESH)) {
                return false;
            }
            //i dont quite understand why igraphicspipeline doesnt require a fragment shader. is it not required by vulkan?
            if (!stagePresence.hasFlags(hlsl::ShaderStage::ESS_FRAGMENT)) {
                return false;
            }

            return true;
        }

        inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }

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