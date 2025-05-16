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
        using base_t = ICPUPipeline<IComputePipeline<ICPUPipelineLayout>>;

    public:

        static core::smart_refctd_ptr<ICPUComputePipeline> create(const ICPUPipelineLayout* layout)
        {
            auto retval = new ICPUComputePipeline(layout);
            return core::smart_refctd_ptr<ICPUComputePipeline>(retval,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_COMPUTE_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
        //!
        inline core::unordered_set<const IAsset*> computeDependants() const override
        {
            return computeDependantsImpl(this);
        }

        inline core::unordered_set<IAsset*> computeDependants() override
        {
            return computeDependantsImpl(this);
        }

        inline std::span<const SShaderSpecInfo> getSpecInfo(hlsl::ShaderStage stage) const override final
        {
            if (stage==hlsl::ShaderStage::ESS_COMPUTE)
                return {&m_specInfo,1};
            return {};
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

        inline core::smart_refctd_ptr<base_t> clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout, uint32_t depth) const override final
        {
            auto newPipeline = new ICPUComputePipeline(layout.get());
            newPipeline->m_specInfo = m_specInfo.clone(depth);
            return core::smart_refctd_ptr<base_t>(newPipeline, core::dont_grab);
        }

        explicit ICPUComputePipeline(const ICPUPipelineLayout* layout):
          base_t(layout, {})
          {}

        template <typename Self>
          requires(std::same_as<std::remove_cv_t<Self>, ICPUComputePipeline>)
        static auto computeDependantsImpl(Self* self) {
            using asset_ptr_t = std::conditional_t<std::is_const_v<Self>, const IAsset*, IAsset*>;
            return core::unordered_set<asset_ptr_t>{ self->m_layout.get(), self->m_specInfo.shader.get() };
        }
};

}
#endif