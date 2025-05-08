
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
        struct SHitGroupSpecInfo {
            SShaderSpecInfo closestHit;
            SShaderSpecInfo anyHit;
            SShaderSpecInfo intersection;

            SHitGroupSpecInfo clone(uint32_t depth) const
            {
                auto newSpecInfo = *this;
                if (depth > 0u)
                {
                    newSpecInfo.closestHit.shader = core::smart_refctd_ptr_static_cast<IShader>(this->closestHit.shader->clone(depth - 1u));
                    newSpecInfo.anyHit.shader = core::smart_refctd_ptr_static_cast<IShader>(this->anyHit.shader->clone(depth - 1u));
                    newSpecInfo.intersection.shader = core::smart_refctd_ptr_static_cast<IShader>(this->intersection.shader->clone(depth - 1u));
                }
                return newSpecInfo;
            }
        };

        static core::smart_refctd_ptr<ICPURayTracingPipeline> create(const ICPUPipelineLayout* layout)
        {
            auto retval = new ICPURayTracingPipeline(layout);
            return core::smart_refctd_ptr<ICPURayTracingPipeline>(retval,core::dont_grab);
        }

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto newPipeline = new ICPURayTracingPipeline(layout.get());
            newPipeline->m_raygen = m_raygen.clone(depth);

            newPipeline->m_misses.resize(m_misses.size());
            for (auto specInfo_i = 0u; specInfo_i < m_misses.size(); specInfo_i++)
            {
                newPipeline->m_misses[specInfo_i] = m_misses[specInfo_i].clone(depth);
            }

            newPipeline->m_hitGroups.resize(m_hitGroups.size());
            for (auto specInfo_i = 0u; specInfo_i < m_misses.size(); specInfo_i++)
            {
                newPipeline->m_hitGroups[specInfo_i] = m_hitGroups[specInfo_i].clone(depth);
            }

            newPipeline->m_callables.resize(m_callables.size());
            for (auto specInfo_i = 0u; specInfo_i < m_callables.size(); specInfo_i++)
            {
                newPipeline->m_callables[specInfo_i] = m_callables[specInfo_i].clone(depth);
            }

            newPipeline->m_params = m_params;
            return core::smart_refctd_ptr<base_t>(newPipeline);
        }

        constexpr static inline auto AssetType = ET_RAYTRACING_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        //!
        inline size_t getDependantCount() const override { 
            //TODO(kevinyu): Implement or refactor the api design to something else
            return 0;
        }

        inline virtual std::span<const SShaderSpecInfo> getSpecInfo(hlsl::ShaderStage stage) const override final
        {
          switch (stage) 
          {
            case hlsl::ShaderStage::ESS_RAYGEN:
              return { &m_raygen, 1 };
          }
            return {};
        }

        inline virtual bool valid() const override final
        {
            // TODO(kevinyu): Fix this temporary dummy code
            return true;
        }

    protected:
        virtual ~ICPURayTracingPipeline() = default;

        inline IAsset* getDependant_impl(const size_t ix) override
        {
            //TODO(kevinyu): remove this function, since this is expensive
            return nullptr;
        }


    private:
        
        SShaderSpecInfo m_raygen;
        core::vector<SShaderSpecInfo> m_misses;
        core::vector<SHitGroupSpecInfo> m_hitGroups;
        core::vector<SShaderSpecInfo> m_callables;

        explicit ICPURayTracingPipeline(const ICPUPipelineLayout* layout)
            : base_t(layout, {})
            {}

};

}
#endif
