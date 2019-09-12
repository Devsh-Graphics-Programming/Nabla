#ifndef __IRR_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __IRR_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"
#include "irr/asset/IPipelineLayout.h"

namespace irr { namespace asset
{

class ICPUPipelineLayout : public IAsset, public IPipelineLayout<ICPUDescriptorSetLayout>
{
public:
    using IPipelineLayout<ICPUDescriptorSetLayout>::IPipelineLayout;

protected:
    virtual ~ICPUPipelineLayout() = default;
};

}}

#endif