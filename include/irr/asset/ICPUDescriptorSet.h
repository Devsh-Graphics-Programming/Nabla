#ifndef __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUBufferView.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/ICPUSampler.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"
#include "irr/asset/IDescriptorSet.h"

namespace irr
{
namespace asset
{

class ICPUDescriptorSet : public IDescriptorSet<ICPUDescriptorSetLayout, ICPUBuffer, ICPUImageView, ICPUBufferView, ICPUSampler>, public IAsset
{
public:
    using base_t = IDescriptorSet<ICPUDescriptorSetLayout, ICPUBuffer, ICPUImageView, ICPUBufferView, ICPUSampler>;

    using base_t::base_t;

    //! Contructor preallocating memory for SDescriptorBindings which user can fill later (using non-const getDescriptors()).
    //! @see getDescriptors()
    ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout) :
        base_t(std::move(_layout))
    {}


    size_t conservativeSizeEstimate() const override { return m_descriptors->size()*sizeof(SDescriptorBinding); }
    void convertToDummyObject() override 
    {
        m_descriptors = nullptr;
        m_bindingToIx = nullptr;
    }
    E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET; }

    ICPUDescriptorSetLayout* getLayout() { return m_layout.get(); }
    core::SRange<SDescriptorBinding> getDescriptors() 
    { 
        return m_descriptors ? core::SRange<SDescriptorBinding>{m_descriptors->begin(), m_descriptors->end()} : core::SRange<SDescriptorBinding>{nullptr, nullptr};
    }

protected:
    virtual ~ICPUDescriptorSet() = default;
};

}}

#endif