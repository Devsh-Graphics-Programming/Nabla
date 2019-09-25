#ifndef __IRR_I_GPU_SAMPLER_H_INCLUDED__
#define __IRR_I_GPU_SAMPLER_H_INCLUDED__

#include "irr/asset/ISampler.h"
#include "irr/core/IReferenceCounted.h"

namespace irr {
namespace video
{

class IGPUSampler : public asset::ISampler, public core::IReferenceCounted
{
protected:
    virtual ~IGPUSampler() = default;

public:
    using asset::ISampler::ISampler;
};

}}

#endif