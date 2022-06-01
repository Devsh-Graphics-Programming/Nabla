#ifndef __NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED__
#define __NBL_I_GPU_GRAPHICS_PIPELINE_H_INCLUDED__


#include "nbl/asset/IGraphicsPipeline.h"

#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPURenderpassIndependentPipeline.h"


namespace nbl::video
{

class NBL_API IGPUGraphicsPipeline : public core::IReferenceCounted, public asset::IGraphicsPipeline<IGPURenderpassIndependentPipeline, IGPURenderpass>, public IBackendObject
{
        using base_t = asset::IGraphicsPipeline<IGPURenderpassIndependentPipeline, IGPURenderpass>;

    protected:
        ~IGPUGraphicsPipeline() = default;

    public:
        IGPUGraphicsPipeline(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : base_t(std::move(params)), IBackendObject(std::move(dev)) {}
};

}

#endif