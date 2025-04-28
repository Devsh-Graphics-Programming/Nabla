// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

//! CPU Version of Compute Pipeline
class ICPUComputePipeline : public ICPUPipeline<IPipeline<ICPUPipelineLayout>>
{
        using base_t = ICPUPipeline<IPipeline<ICPUPipelineLayout>>;

    public:
        struct SCreationParams final : IPipeline<ICPUPipelineLayout>::SCreationParams
        {
            IPipelineBase::SShaderSpecInfo<true> shader;
        };
        static core::smart_refctd_ptr<ICPUComputePipeline> create(const SCreationParams& params)
        {
            if (!params.layout)
                return nullptr;
            auto retval = new ICPUComputePipeline(core::smart_refctd_ptr<const ICPUPipelineLayout>(params.layout));
            
            if (!retval->setSpecInfo(params.shader))
            {
                retval->drop();
                return nullptr;
            }
            return core::smart_refctd_ptr<ICPUComputePipeline>(retval,core::dont_grab);
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override final
        {
            core::smart_refctd_ptr<ICPUPipelineLayout> layout;
            if (_depth>0u && m_layout)
                layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u));

            auto cp = new ICPUComputePipeline(std::move(layout));
            if (m_specInfo.shader)
            {
                SShaderSpecInfo<true> specInfo = m_specInfo;
                if (_depth > 0u)
                {
                  specInfo.shader = core::smart_refctd_ptr_static_cast<IShader>(m_specInfo.shader->clone(_depth - 1u));
                }
                cp->setSpecInfo(specInfo);
            }
            return core::smart_refctd_ptr<ICPUComputePipeline>(cp,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_COMPUTE_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }
        
		//!
		inline size_t getDependantCount() const override {return 2;}

    protected:
        using base_t::base_t;
        virtual ~ICPUComputePipeline() = default;

		inline IAsset* getDependant_impl(const size_t ix) override
        {
            if (ix!=0)
                return m_specInfo.shader.get();
            return const_cast<ICPUPipelineLayout*>(m_layout.get());
        }

        inline bool setSpecInfo(const IPipelineBase::SShaderSpecInfo<true>& info)
        {
          const auto specSize = info.valid();
          if (specSize < 0) return false;
          if (info.stage != hlsl::ESS_COMPUTE) return false;
          m_specInfo = info;
          return true;
        }

    private:
        SShaderSpecInfo<true> m_specInfo;

};

}
#endif