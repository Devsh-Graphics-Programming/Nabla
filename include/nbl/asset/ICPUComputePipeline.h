// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/ICPUPipeline.h"
#include "nbl/asset/IComputePipeline.h"


namespace nbl::asset
{

//! CPU Version of Compute Pipeline
class ICPUComputePipeline final : public ICPUPipeline<IComputePipeline<ICPUPipelineLayout>>
{
        using pipeline_base_t = IComputePipeline<ICPUPipelineLayout>;
        using base_t = ICPUPipeline<IComputePipeline<ICPUPipelineLayout>>;

    public:

        static core::smart_refctd_ptr<ICPUComputePipeline> create(ICPUPipelineLayout* layout)
        {
            auto retval = new ICPUComputePipeline(layout);
            return core::smart_refctd_ptr<ICPUComputePipeline>(retval,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_COMPUTE_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }

        inline std::span<const SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage) const override
        {
            if (stage==hlsl::ShaderStage::ESS_COMPUTE)
                return {&m_specInfo,1};
            return {};
        }

        inline std::span<SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage)
        {
            return base_t::getSpecInfos(stage);
        }

        inline SShaderSpecInfo& getSpecInfo()
        {
            return m_specInfo;
        }

        inline const SShaderSpecInfo& getSpecInfo() const
        {
            return m_specInfo;
        }

        inline const SCachedCreationParams& getCachedCreationParams() const
        {
            return pipeline_base_t::getCachedCreationParams();
        }

        inline SCachedCreationParams& getCachedCreationParams()
        {
            assert(isMutable());
            return m_params;
        }

        inline bool valid() const override
        {
            if (!m_layout) return false;
            if (!m_layout->valid()) return false;
            return m_specInfo.valid();
        }

    protected:
        using base_t::base_t;
        virtual ~ICPUComputePipeline() = default;


    private:
        SShaderSpecInfo m_specInfo;

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto newPipeline = new ICPUComputePipeline(layout.get());
            newPipeline->m_specInfo = m_specInfo.clone(depth);
            return core::smart_refctd_ptr<base_t>(newPipeline, core::dont_grab);
        }

        explicit ICPUComputePipeline(ICPUPipelineLayout* layout):
          base_t(layout, {})
          {}
        
        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const override
        {
            if (!visit(m_layout.get())) return;
            if (!visit(m_specInfo.shader.get())) return;
        }
};

}
#endif