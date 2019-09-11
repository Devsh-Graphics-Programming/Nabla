#ifndef __IRR_I_GPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_GPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IDescriptorSet.h"
#include "IGPUBuffer.h"
#include "ITexture.h"

namespace irr { namespace video
{

class IGPUDescriptorSet : public asset::IDescriptorSet<IGPUBuffer, ITexture>, public core::IReferenceCounted
{
public:
    using asset::IDescriptorSet<IGPUBuffer, ITexture>::IDescriptorSet;

protected:
    virtual ~IGPUDescriptorSet() = default;
};

}}

#endif