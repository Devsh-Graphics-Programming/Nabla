#ifndef __IRR_I_SAMPLER_H_INCLUDED__
#define __IRR_I_SAMPLER_H_INCLUDED__

#include "irr/asset/SSamplerParams.h"

namespace irr {
namespace asset
{

class ISampler
{
protected:
    ISampler(const SSamplerParams& _params) : m_params(_params) {}
    virtual ~ISampler() = default;

    SSamplerParams m_params;

public:
    const SSamplerParams& getParams() const { return m_params; }
};

}}

#endif 