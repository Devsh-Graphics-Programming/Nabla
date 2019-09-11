#ifndef __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IDescriptorSet.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUTexture.h"
#include "irr/asset/ICPUBuffer.h"

namespace irr { namespace asset
{

class ICPUDescriptorSet : public IDescriptorSet<ICPUBuffer, ICPUTexture>, public IAsset
{
public:
    using IDescriptorSet<ICPUBuffer, ICPUTexture>::IDescriptorSet;

protected:
    virtual ~ICPUDescriptorSet() = default;
};

}}

#endif