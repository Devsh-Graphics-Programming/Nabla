// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_SAMPLER_H_INCLUDED__
#define __NBL_I_GPU_SAMPLER_H_INCLUDED__

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