// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/ICPUPipelineLayout.h"


namespace nbl::asset
{

// Common Base class for pipelines
template<typename PipelineNonAssetBase>
class ICPUPipeline : public IAsset, public PipelineNonAssetBase
{
        using this_t = ICPUPipeline<PipelineNonAssetBase>;
        using shader_info_spec_t = IPipelineBase::SShaderSpecInfo<true>;

    public:

        // extras for this class
        ICPUPipelineLayout* getLayout() 
        {
            assert(isMutable());
            return const_cast<ICPUPipelineLayout*>(PipelineNonAssetBase::m_layout.get());
        }
        const ICPUPipelineLayout* getLayout() const { return PipelineNonAssetBase::m_layout.get(); }

        inline void setLayout(core::smart_refctd_ptr<const ICPUPipelineLayout>&& _layout)
        {
            assert(isMutable());
            PipelineNonAssetBase::m_layout = std::move(_layout);
        }

    protected:
        using PipelineNonAssetBase::PipelineNonAssetBase;
        virtual ~ICPUPipeline() = default;

};

}
#endif