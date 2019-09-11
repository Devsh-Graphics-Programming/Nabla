#ifndef __IRR_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __IRR_I_GPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/IPipelineLayout.h"
#include "irr/video/IGPUDescriptorSetLayout.h"

namespace irr {
namespace video
{

class IGPUPipelineLayout : public core::IReferenceCounted, public asset::IPipelineLayout<IGPUDescriptorSetLayout>
{
protected:
    virtual ~IGPUPipelineLayout() = default;
};

}
}

#endif