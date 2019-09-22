#ifndef __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IDescriptorSet.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUTexture.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ICPUBufferView.h"

namespace irr { namespace asset
{

class ICPUDescriptorSet : public IDescriptorSet<ICPUBuffer, ICPUTexture, ICPUBufferView>, public IAsset
{
public:
    using IDescriptorSet<ICPUBuffer, ICPUTexture, ICPUBufferView>::IDescriptorSet;

    size_t conservativeSizeEstimate() const override { return m_descriptors->size()*sizeof(SWriteDescriptorSet); }
    void convertToDummyObject() override { m_descriptors = nullptr; }
    E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET; }

protected:
    virtual ~ICPUDescriptorSet() = default;
};

}}

#endif