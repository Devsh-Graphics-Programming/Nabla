// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/asset/IGraphicsPipeline.h"
#include "nbl/asset/ICPURenderpass.h"
#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

class ICPUGraphicsPipeline final : public ICPUPipeline<IGraphicsPipeline<ICPUPipelineLayout,ICPURenderpass>>
{
        using pipeline_base_t = IGraphicsPipeline<ICPUPipelineLayout, ICPURenderpass>;
        using base_t = ICPUPipeline<pipeline_base_t>;

    public:
        explicit ICPUGraphicsPipeline(const ICPUPipelineLayout* layout)
            : base_t(layout, {}, {})
            {}
        
        static core::smart_refctd_ptr<ICPUGraphicsPipeline> create(const ICPUPipelineLayout* layout)
        {
            auto retval = new ICPUGraphicsPipeline(layout);
            return core::smart_refctd_ptr<ICPUGraphicsPipeline>(retval,core::dont_grab);
        }

        inline base_t* clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto* newPipeline = new ICPUGraphicsPipeline(layout.get());
            for (auto i = 0; i < GRAPHICS_SHADER_STAGE_COUNT; i++)
                newPipeline->m_specInfos[i] = m_specInfos[i];
            newPipeline->m_params = m_params;
            newPipeline->m_renderpass = m_renderpass;
            
            for (auto specInfo_i = 0u; specInfo_i < m_specInfos.size(); specInfo_i++)
            {
                newPipeline->m_specInfos[specInfo_i] = newPipeline->cloneSpecInfo(m_specInfos[specInfo_i], depth);
            }

            return newPipeline;
        }

        constexpr static inline auto AssetType = ET_GRAPHICS_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        inline size_t getDependantCount() const override
        {
            auto stageCount = 2; // the layout and renderpass
            for (const auto& info : m_specInfos)
            {
              if (info.shader)
                stageCount++;
            }
            return stageCount;
        }

        inline SCachedCreationParams& getCachedCreationParams()
        {
            assert(isMutable());
            return m_params;
        }

        inline virtual std::span<SShaderSpecInfo> getSpecInfo(hlsl::ShaderStage stage) override final
        {
            const auto stageIndex = stageToIndex(stage);
            if (isMutable() && stageIndex != -1)
            {
                return { &m_specInfos[stageIndex], 1 };
            }
            return {};
        }

        inline virtual bool valid() const override final
        {
            // TODO(kevinyu): Fix this temporary stub code
            return true;
        }

    protected:
        using base_t::base_t;
        ~ICPUGraphicsPipeline() = default;

        inline IAsset* getDependant_impl(const size_t ix) override
        {
            if (ix==0)
                return const_cast<ICPUPipelineLayout*>(m_layout.get());
            if (ix==1)
                return m_renderpass.get();
            size_t stageCount = 0;
            for (auto& specInfo : m_specInfos)
            {
                if (specInfo.shader)
                    if ((stageCount++)==ix-2) return specInfo.shader.get();
            }
            return nullptr;
        }

        inline int8_t stageToIndex(const hlsl::ShaderStage stage) const
        {
            const auto stageIx = hlsl::findLSB(stage);
            if (stageIx<0 || stageIx>= GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
              return -1;
            return stageIx;
        }

        std::array<SShaderSpecInfo, GRAPHICS_SHADER_STAGE_COUNT> m_specInfos;
};

}

#endif