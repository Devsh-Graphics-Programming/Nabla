// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_I_CPU_RASTERIZATION_PIPELINE_H_INCLUDED_
#define _NBL_I_CPU_RASTERIZATION_PIPELINE_H_INCLUDED_

//i dont think ICPURasterizationPipeline is reasonable -corey
//ICPUGraphicsPipeline needs to use ICPUPipeline<IGraphicsPipeline>, but ICPURaster would need ICPUPipeline<IRasterizationPipeline>
//skipping IGraphics would NOT be good, some data would be lost, 
//and templating ICPURaster for IGraphics would defeat the purpose of polymorphism


#include "nbl/asset/IRasterizationPipeline.h"
#include "nbl/asset/ICPURenderpass.h"
#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

class ICPURasterizationPipeline : public ICPUPipeline<IRasterizationPipeline<ICPUPipelineLayout,ICPURenderpass>>
{
        using pipeline_base_t = IRasterizationPipeline<ICPUPipelineLayout, ICPURenderpass>;
        using base_t = ICPUPipeline<pipeline_base_t>;

    public:
        
        static core::smart_refctd_ptr<IRasterizationPipeline> create(ICPUPipelineLayout* layout, ICPURenderpass* renderpass = nullptr)
        {
            auto retval = new ICPURasterizationPipeline(layout, renderpass);
            return core::smart_refctd_ptr<ICPURasterizationPipeline>(retval,core::dont_grab);
        }

    protected:
        using base_t::base_t;
        virtual ~ICPURasterizationPipeline() override = default;

    private:
        explicit ICPURasterizationPipeline(ICPUPipelineLayout* layout, ICPURenderpass* renderpass)
            : base_t(layout, {}, renderpass)
            {}
};

}

#endif