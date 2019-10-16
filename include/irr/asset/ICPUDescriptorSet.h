#ifndef __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IDescriptorSet.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUTexture.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ICPUBufferView.h"
#include "irr/asset/ICPUSampler.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"

namespace irr { namespace asset
{

class ICPUDescriptorSet : public IDescriptorSet<ICPUDescriptorSetLayout, ICPUBuffer, ICPUTexture, ICPUBufferView, ICPUSampler>, public IAsset
{
public:
    using base_t = IDescriptorSet<ICPUDescriptorSetLayout, ICPUBuffer, ICPUTexture, ICPUBufferView, ICPUSampler>;

    using base_t::base_t;

    //! Contructor preallocating memory for `_descriptorCount` of SWriteDescriptorSets which user can fill later (using non-const getDescriptors()).
    //! @see getDescriptors()
    ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout, size_t _descriptorCount) :
        base_t(std::move(_layout), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SWriteDescriptorSet>>(_descriptorCount))
    {}


    size_t conservativeSizeEstimate() const override { return m_descriptors->size()*sizeof(SWriteDescriptorSet); }
    void convertToDummyObject() override { m_descriptors = nullptr; }
    E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET; }

    ICPUDescriptorSetLayout* getLayout() { return m_layout.get(); }
    core::SRange<SWriteDescriptorSet> getDescriptors() { return { m_descriptors->begin(), m_descriptors->end() }; }

protected:
    virtual ~ICPUDescriptorSet() = default;
};

}}

#endif