#ifndef __IRR_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/IDescriptorSetLayout.h"
#include "irr/asset/IAsset.h"

namespace irr { namespace asset
{

class ICPUDescriptorSetLayout : public IDescriptorSetLayout, public IAsset
{
public:
    using IDescriptorSetLayout::IDescriptorSetLayout;

protected:
    virtual ~ICPUDescriptorSetLayout() = default;
};

}}

#endif