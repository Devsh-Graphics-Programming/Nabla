#ifndef __IRR_I_CPU_COMPUTE_PIPELINE_H_INCLUDED__
#define __IRR_I_CPU_COMPUTE_PIPELINE_H_INCLUDED__

#include "irr/asset/IComputePipeline.h"
#include "irr/asset/ICPUPipelineLayout.h"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/asset/IAsset.h"

namespace irr {
namespace asset
{

class ICPUComputePipeline : public IComputePipeline<ICPUSpecializedShader, ICPUPipelineLayout>, public IAsset
{
    using base_t = IComputePipeline<ICPUSpecializedShader, ICPUPipelineLayout>;

public:
    using base_t::base_t;

    size_t conservativeSizeEstimate() const override { return sizeof(base_t); }
    void convertToDummyObject() override { }
    E_TYPE getAssetType() const override { return ET_COMPUTE_PIPELINE; }

protected:
    virtual ~ICPUComputePipeline() = default;
};

}}

#endif