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

    size_t conservativeSizeEstimate() const override { return m_bindings->size()*sizeof(SBinding) + m_samplers->size()*sizeof(SSamplerParams); }
    void convertToDummyObject() override
    {
        m_bindings = nullptr;
        m_samplers = nullptr;
    }
    E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET_LAYOUT; }

protected:
    virtual ~ICPUDescriptorSetLayout() = default;
};

}}

#endif