#ifndef __IRR_I_GPU_SAMPLER_H_INCLUDED__
#define __IRR_I_GPU_SAMPLER_H_INCLUDED__

#include "irr/asset/ISampler.h"

namespace irr
{
namespace video
{

class IGPUSampler : public asset::ISampler
{
protected:
    virtual ~IGPUSampler() = default;

public:
    using asset::ISampler::ISampler;
};

}
}

#endif