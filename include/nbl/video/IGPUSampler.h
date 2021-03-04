// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SAMPLER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SAMPLER_H_INCLUDED__

#include "nbl/asset/ISampler.h"
#include "nbl/video/IBackendObject.h"

namespace nbl
{
namespace video
{

class IGPUSampler : public asset::ISampler, public IBackendObject
{
protected:
    virtual ~IGPUSampler() = default;

public:
    IGPUSampler(ILogicalDevice* dev, const SParams& params) : ISampler(params), IBackendObject(dev) {}
};

}
}

#endif