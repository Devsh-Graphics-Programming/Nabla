#ifndef _NBL_I_CPU_MESH_PIPELINE_H_INCLUDED_
#define _NBL_I_CPU_MESH_PIPELINE_H_INCLUDED_


#include "nbl/asset/IMeshPipeline.h"
#include "nbl/asset/ICPURenderpass.h"
#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

class ICPUMeshPipeline final : public ICPUPipeline<IMeshPipeline<ICPUPipelineLayout,ICPURenderpass>>
{
        using pipeline_base_t = IMeshPipeline<ICPUPipelineLayout, ICPURenderpass>;
        using base_t = ICPUPipeline<pipeline_base_t>;

    public:
        
        static core::smart_refctd_ptr<ICPUMeshPipeline> create(ICPUPipelineLayout* layout, ICPURenderpass* renderpass = nullptr)
        {
            auto retval = new ICPUMeshPipeline(layout, renderpass);
            return core::smart_refctd_ptr<ICPUMeshPipeline>(retval,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_MESH_PIPELINE;
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

        inline std::span<const SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage) const override final
        {
            switch (stage) {
                case hlsl::ShaderStage::ESS_TASK:       return { &m_specInfos[0], 1 };
                case hlsl::ShaderStage::ESS_MESH:       return { &m_specInfos[1], 1 };
                case hlsl::ShaderStage::ESS_FRAGMENT:   return { &m_specInfos[2], 1 };
            }
            return {};
        }

        inline std::span<SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage)
        {
            return base_t::getSpecInfos(stage);
        }

        SShaderSpecInfo* getSpecInfo(const hlsl::ShaderStage stage)
        {
            if (!isMutable()) return nullptr;
            switch (stage) {
                case hlsl::ShaderStage::ESS_TASK:       return &m_specInfos[0];
                case hlsl::ShaderStage::ESS_MESH:       return &m_specInfos[1];
                case hlsl::ShaderStage::ESS_FRAGMENT:   return &m_specInfos[2];
            }
            return nullptr;
        }

        const SShaderSpecInfo* getSpecInfo(const hlsl::ShaderStage stage) const
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
            return hasRequiredStages(stagePresence);
        }

    protected:
        using base_t::base_t;
        virtual ~ICPUMeshPipeline() override = default;

        std::array<SShaderSpecInfo, MESH_SHADER_STAGE_COUNT> m_specInfos;

    private:
        explicit ICPUMeshPipeline(ICPUPipelineLayout* layout, ICPURenderpass* renderpass)
            : base_t(layout, {}, renderpass)
            {}

        static inline int8_t stageToIndex(const hlsl::ShaderStage stage)
        {
            const auto stageIx = hlsl::findLSB(stage);
            if (stageIx < 0 || stageIx >= MESH_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
              return -1;
            return stageIx;
        }

        static inline hlsl::ShaderStage indexToStage(const int8_t index)
        {
            switch (index) {
                case 0: return hlsl::ShaderStage::ESS_TASK;
                case 1: return hlsl::ShaderStage::ESS_MESH;
                case 2: return hlsl::ShaderStage::ESS_FRAGMENT;
            }
            return hlsl::ShaderStage::ESS_UNKNOWN;
        }

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto* newPipeline = new ICPUMeshPipeline(layout.get(), m_renderpass.get());
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