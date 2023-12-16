#ifndef _NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/asset/IGraphicsPipeline.h"

#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPURenderpassIndependentPipeline.h"


namespace nbl::video
{

class IGPUGraphicsPipeline : public IBackendObject, public asset::IGraphicsPipeline<const IGPURenderpassIndependentPipeline,const IGPURenderpass>
{
        using base_t = asset::IGraphicsPipeline<const IGPURenderpassIndependentPipeline,const IGPURenderpass>;

    protected:
        ~IGPUGraphicsPipeline() = default;

    public:
        IGPUGraphicsPipeline(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : IBackendObject(std::move(dev)), base_t(std::move(params)) {}
};

}

#endif