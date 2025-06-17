
// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IRayTracingPipeline.h"
#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

//! CPU Version of RayTracing Pipeline
class ICPURayTracingPipeline final : public ICPUPipeline<IRayTracingPipeline<ICPUPipelineLayout>>
{
        using pipeline_base_t = IRayTracingPipeline<ICPUPipelineLayout>;
        using base_t = ICPUPipeline<pipeline_base_t>;

    public:
        struct SHitGroupSpecInfos {
            core::vector<SShaderSpecInfo> closestHits;
            core::vector<SShaderSpecInfo> anyHits;
            core::vector<SShaderSpecInfo> intersections;
        };

        static core::smart_refctd_ptr<ICPURayTracingPipeline> create(const ICPUPipelineLayout* layout)
        {
            auto retval = new ICPURayTracingPipeline(layout);
            return core::smart_refctd_ptr<ICPURayTracingPipeline>(retval,core::dont_grab);
        }

        

        constexpr static inline auto AssetType = ET_RAYTRACING_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        inline std::span<const SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage) const override final
        {
            switch (stage) 
            {
                case hlsl::ShaderStage::ESS_RAYGEN:
                  return { &m_raygen, 1 };
                case hlsl::ShaderStage::ESS_MISS:
                  return m_misses;
                case hlsl::ShaderStage::ESS_ANY_HIT:
                  return m_hitGroups.anyHits;
                case hlsl::ShaderStage::ESS_CLOSEST_HIT:
                  return m_hitGroups.closestHits;
                case hlsl::ShaderStage::ESS_INTERSECTION:
                  return m_hitGroups.intersections;
                case hlsl::ShaderStage::ESS_CALLABLE:
                  return m_callables;

            }
            return {};
        }

        inline std::span<SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage)
        {
            return base_t::getSpecInfos(stage);
        }

        inline core::vector<SShaderSpecInfo>* getSpecInfoVector(const hlsl::ShaderStage stage)
        {
            if (!isMutable()) return nullptr;
            switch (stage) 
            {
                // raygen is not stored as vector so we can't return it here. Use getSpecInfo
                case hlsl::ShaderStage::ESS_MISS:
                  return &m_misses;
                case hlsl::ShaderStage::ESS_ANY_HIT:
                  return &m_hitGroups.anyHits;
                case hlsl::ShaderStage::ESS_CLOSEST_HIT:
                  return &m_hitGroups.closestHits;
                case hlsl::ShaderStage::ESS_INTERSECTION:
                  return &m_hitGroups.intersections;
                case hlsl::ShaderStage::ESS_CALLABLE:
                  return &m_callables;

            }
            return nullptr;
        }


        inline bool valid() const override final
        {
            if (!m_layout) return false;
            if (!m_layout->valid()) return false;
            if (m_raygen.valid() == SShaderSpecInfo::INVALID_SPEC_INFO) return false;
            return true;
        }

        inline const SCachedCreationParams& getCachedCreationParams() const
        {
            return pipeline_base_t::getCachedCreationParams();
        }

        inline SCachedCreationParams& getCachedCreationParams() {
            assert(isMutable());
            return m_params;
        }

    protected:
        virtual ~ICPURayTracingPipeline() = default;

    private:
        
        SShaderSpecInfo m_raygen;
        core::vector<SShaderSpecInfo> m_misses;
        SHitGroupSpecInfos m_hitGroups;
        core::vector<SShaderSpecInfo> m_callables;

        explicit ICPURayTracingPipeline(const ICPUPipelineLayout* layout)
            : base_t(layout, {})
            {}

        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const override
        {
            if (!visit(m_raygen.shader.get()) return;
            for (const auto& missInfo : self->m_misses) if (!visit(missInfo.shader.get())) return;
            for (const auto& anyHitInfo : self->m_hitGroups.anyHits) if (!visit(anyHitInfo.shader.get())) return;
            for (const auto& closestHitInfo : self->m_hitGroups.closestHits) if (!visit(closestHitInfo.shader.get())) return;
            for (const auto& intersectionInfo : self->m_hitGroups.intersections) if (!visit(intersectionInfo.shader.get())) return;
            for (const auto& callableInfo : self->m_callables) if(!visit(callableInfo.shader.get())) return;
        }

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto newPipeline = new ICPURayTracingPipeline(layout.get());
            newPipeline->m_raygen = m_raygen.clone(depth);

            auto cloneSpecInfos = [depth](const core::vector<SShaderSpecInfo>& specInfos) -> core::vector<SShaderSpecInfo> {
                core::vector<SShaderSpecInfo> results;
                results.resize(specInfos.size());
                for (auto specInfo_i = 0u; specInfo_i < specInfos.size(); specInfo_i++)
                    results[specInfo_i] = specInfos[specInfo_i].clone(depth);
                return results;
            };
            newPipeline->m_misses = cloneSpecInfos(m_misses);
            newPipeline->m_hitGroups.anyHits = cloneSpecInfos(m_hitGroups.anyHits);
            newPipeline->m_hitGroups.closestHits = cloneSpecInfos(m_hitGroups.closestHits);
            newPipeline->m_hitGroups.intersections = cloneSpecInfos(m_hitGroups.intersections);
            newPipeline->m_callables = cloneSpecInfos(m_callables);

            newPipeline->m_params = m_params;
            return core::smart_refctd_ptr<base_t>(newPipeline);
        }
};

}
#endif
