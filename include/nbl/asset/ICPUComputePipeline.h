// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

//! CPU Version of Compute Pipeline
class ICPUComputePipeline : public ICPUPipeline<IPipeline<ICPUPipelineLayout>,1>
{
        using base_t = ICPUPipeline<IPipeline<ICPUPipelineLayout>,1>;

    public:
        struct SCreationParams final : IPipeline<ICPUPipelineLayout>::SCreationParams
        {
            ICPUShader::SSpecInfo shader;
        };
        static core::smart_refctd_ptr<ICPUComputePipeline> create(const SCreationParams& params)
        {
            if (!params.layout)
                return nullptr;
            auto retval = new ICPUComputePipeline(core::smart_refctd_ptr<ICPUPipelineLayout>(params.layout));
            if (!retval->setSpecInfo(params.shader))
            {
                retval->drop();
                return nullptr;
            }
            return core::smart_refctd_ptr<ICPUComputePipeline>(retval,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_COMPUTE_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }

        // provide default arg
        inline IShader::SSpecInfo<ICPUShader> getSpecInfo() {return base_t::getSpecInfo(ICPUShader::ESS_COMPUTE);}
        inline IShader::SSpecInfo<const ICPUShader> getSpecInfo() const {return base_t::getSpecInfo(ICPUShader::ESS_COMPUTE);}

    protected:
        using base_t::base_t;
        virtual ~ICPUComputePipeline() = default;

        base_t* clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout) const override
        {
            return new ICPUComputePipeline(std::move(layout));
        }
        inline int8_t stageToIndex(const ICPUShader::E_SHADER_STAGE stage) const override
        {
            return stage!=ICPUShader::ESS_COMPUTE ? (-1):0;
        }
};

}
#endif