#ifndef __IRR_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/asset/IRenderpassIndependentPipeline.h"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/asset/ICPUPipelineLayout.h"

namespace irr {
namespace asset
{

class ICPURenderpassIndependentPipeline : public IRenderpassIndependentPipeline<ICPUSpecializedShader, ICPUPipelineLayout>, public IAsset
{
    using base_t = IRenderpassIndependentPipeline<ICPUSpecializedShader, ICPUPipelineLayout>;

public:
    using base_t::base_t;

    //maybe setters (shaders, layout, other params) for CPU counterpart only

protected:
    virtual ~ICPURenderpassIndependentPipeline() = default;
};

}}

#endif