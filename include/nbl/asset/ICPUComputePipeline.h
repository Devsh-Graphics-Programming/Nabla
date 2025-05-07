// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

//! CPU Version of Compute Pipeline
class ICPUComputePipeline final : public ICPUPipeline<IPipeline<ICPUPipelineLayout>>
{
        using base_t = ICPUPipeline<IPipeline<ICPUPipelineLayout>>;

    public:
        explicit ICPUComputePipeline(const ICPUPipelineLayout* layout):
          base_t(core::smart_refctd_ptr<ICPUPipelineLayout>(layout))
          {}

        static core::smart_refctd_ptr<ICPUComputePipeline> create(const ICPUPipelineLayout* layout)
        {
            auto retval = new ICPUComputePipeline(layout);
            return core::smart_refctd_ptr<ICPUComputePipeline>(retval,core::dont_grab);
        }

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto newPipeline = new ICPUComputePipeline(std::move(layout));
            newPipeline->m_specInfo = m_specInfo.clone(depth);
            return core::smart_refctd_ptr<base_t>(newPipeline, core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_COMPUTE_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        //!
        inline size_t getDependantCount() const override { return 2; }

        inline virtual std::span<SShaderSpecInfo> getSpecInfo(hlsl::ShaderStage stage) override final
        {
            if (stage==hlsl::ShaderStage::ESS_COMPUTE && isMutable())
                return {&m_specInfo,1};
            return {};
        }

        inline virtual bool valid() const override final
        {
            return m_specInfo.valid();
        }

    protected:
        using base_t::base_t;
        virtual ~ICPUComputePipeline() = default;

        inline IAsset* getDependant_impl(const size_t ix) override
        {
            if (ix!=0)
                return m_specInfo.shader.get();
            return const_cast<ICPUPipelineLayout*>(m_layout.get());
        }


    private:
        SShaderSpecInfo m_specInfo;

};

}
#endif