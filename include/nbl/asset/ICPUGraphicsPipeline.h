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
        
        static core::smart_refctd_ptr<ICPUGraphicsPipeline> create(const ICPUPipelineLayout* layout)
        {
            auto retval = new ICPUGraphicsPipeline(layout);
            return core::smart_refctd_ptr<ICPUGraphicsPipeline>(retval,core::dont_grab);
        }

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto* newPipeline = new ICPUGraphicsPipeline(layout.get());
            newPipeline->m_params = m_params;
            newPipeline->m_renderpass = m_renderpass;
            
            for (auto specInfo_i = 0u; specInfo_i < m_specInfos.size(); specInfo_i++)
            {
                newPipeline->m_specInfos[specInfo_i] = m_specInfos[specInfo_i].clone(depth);
            }

            return core::smart_refctd_ptr<base_t>(newPipeline, core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_GRAPHICS_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        inline core::unordered_set<const IAsset*> computeDependants() const override
        {
            return computeDependantsImpl(this);
        }

        inline core::unordered_set<IAsset*> computeDependants() override
        {
            return computeDependantsImpl(this);
        }

        inline SCachedCreationParams& getCachedCreationParams()
        {
            assert(isMutable());
            return m_params;
        }

        inline virtual std::span<const SShaderSpecInfo> getSpecInfo(hlsl::ShaderStage stage) const override final
        {
            const auto stageIndex = stageToIndex(stage);
            if (stageIndex != -1)
                return { &m_specInfos[stageIndex], 1 };
            return {};
        }


        inline virtual bool valid() const override final
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
        explicit ICPUGraphicsPipeline(const ICPUPipelineLayout* layout)
            : base_t(layout, {}, {})
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

        template <typename Self>
          requires(std::same_as<std::remove_cv_t<Self>, ICPUGraphicsPipeline>)
        static auto computeDependantsImpl(Self* self) {
            using asset_ptr_t = std::conditional_t<std::is_const_v<Self>, const IAsset*, IAsset*>;
            core::unordered_set<asset_ptr_t> dependants = { self->m_layout.get(), self->m_renderpass.get()};
            for (const auto& info : self->m_specInfos)
              if (info.shader) dependants.insert(info.shader.get());
            return dependants;
        }
};

}

#endif