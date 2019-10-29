#ifndef __IRR_I_GPU_COMPUTE_PIPELINE_H_INCLUDED__
#define __IRR_I_GPU_COMPUTE_PIPELINE_H_INCLUDED__

#include "irr/asset/IComputePipeline.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "irr/video/IGPUPipelineLayout.h"

namespace irr {
namespace video
{

class IGPUComputePipeline : public asset::IComputePipeline<IGPUSpecializedShader, IGPUPipelineLayout>, public core::IReferenceCounted
{
    using base_t = asset::IComputePipeline<IGPUSpecializedShader, IGPUPipelineLayout>;

public:
    using base_t::base_t;

protected:
    virtual ~IGPUComputePipeline() = default;

    bool m_allowDispatchBase;
};

}}

#endif