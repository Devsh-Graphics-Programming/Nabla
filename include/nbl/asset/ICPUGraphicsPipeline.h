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
        
        static core::smart_refctd_ptr<ICPUGraphicsPipeline> create(ICPUPipelineLayout* layout, ICPURenderpass* renderpass = nullptr)
        {
            auto retval = new ICPUGraphicsPipeline(layout, renderpass);
            return core::smart_refctd_ptr<ICPUGraphicsPipeline>(retval,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_GRAPHICS_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        inline const SCachedCreationParams& getCachedCreationParams() const
        {
            return pipeline_base_t::getCachedCreationParams();
        }

        inline SCachedCreationParams& getCachedCreationParams()
        {
            assert(isMutable());
            return m_params;
        }

        inline std::span<const SShaderSpecInfo> getSpecInfos(hlsl::ShaderStage stage) const override final
        {
            const auto stageIndex = stageToIndex(stage);
            if (stageIndex != -1)
                return { &m_specInfos[stageIndex], 1 };
            return {};
        }

        inline std::span<SShaderSpecInfo> getSpecInfos(hlsl::ShaderStage stage)
        {
            return base_t::getSpecInfos(stage);
        }

        SShaderSpecInfo* getSpecInfo(hlsl::ShaderStage stage)
        {
            if (!isMutable()) return nullptr;
            const auto stageIndex = stageToIndex(stage);
            if (stageIndex != -1)
                return &m_specInfos[stageIndex];
            return nullptr;
        }

        const SShaderSpecInfo* getSpecInfo(hlsl::ShaderStage stage) const
        {
            const auto stageIndex = stageToIndex(stage);
            if (stageIndex != -1)
                return &m_specInfos[stageIndex];
            return nullptr;
        }

        inline bool valid() const override
        {
            if (!m_layout) return false;
            if (!m_layout->valid())return false;

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576
            if (!m_renderpass || m_params.subpassIx >= m_renderpass->getSubpassCount()) return false;
            
            core::bitflag<hlsl::ShaderStage> stagePresence = {};
            for (auto shader_i = 0u; shader_i < m_specInfos.size(); shader_i++)
            {
                const auto& info = m_specInfos[shader_i];
                if (info.shader)
                    stagePresence |= indexToStage(shader_i);
            }
            return hasRequiredStages(stagePresence, m_params.primitiveAssembly.primitiveType);
        }

    protected:
        using base_t::base_t;
        virtual ~ICPUGraphicsPipeline() override = default;

        std::array<SShaderSpecInfo, GRAPHICS_SHADER_STAGE_COUNT> m_specInfos;

    private:
        explicit ICPUGraphicsPipeline(ICPUPipelineLayout* layout, ICPURenderpass* renderpass)
            : base_t(layout, {}, renderpass)
            {}

        static inline int8_t stageToIndex(const hlsl::ShaderStage stage)
        {
            const auto stageIx = hlsl::findLSB(stage);
            if (stageIx < 0 || stageIx >= GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
              return -1;
            return stageIx;
        }

        static inline hlsl::ShaderStage indexToStage(const int8_t index)
        {
            if (index < 0 || index > GRAPHICS_SHADER_STAGE_COUNT)
                return hlsl::ShaderStage::ESS_UNKNOWN;
            return static_cast<hlsl::ShaderStage>(hlsl::ShaderStage::ESS_VERTEX + index);
        }

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto* newPipeline = new ICPUGraphicsPipeline(layout.get(), m_renderpass.get());
            newPipeline->m_params = m_params;
            
            for (auto specInfo_i = 0u; specInfo_i < m_specInfos.size(); specInfo_i++)
            {
                newPipeline->m_specInfos[specInfo_i] = m_specInfos[specInfo_i].clone(depth);
            }

            return core::smart_refctd_ptr<base_t>(newPipeline, core::dont_grab);
        }

        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const override
        {
            if (!visit(m_layout.get())) return;
            if (!visit(m_renderpass.get())) return;
            for (const auto& info : m_specInfos)
              if (!visit(info.shader.get())) return;
        }
};

}

#endif