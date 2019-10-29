#ifndef __IRR_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/asset/IRenderpassIndependentPipeline.h"
#include "irr/core/IReferenceCounted.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "irr/video/IGPUPipelineLayout.h"

namespace irr {
namespace video
{

class IGPURenderpassIndependentPipeline : public asset::IRenderpassIndependentPipeline<IGPUSpecializedShader, IGPUPipelineLayout>
{
    using base_t = asset::IRenderpassIndependentPipeline<IGPUSpecializedShader, IGPUPipelineLayout>;

public:
    using base_t::base_t;

protected:
    virtual ~IGPURenderpassIndependentPipeline() = default;
};

}}

#endif