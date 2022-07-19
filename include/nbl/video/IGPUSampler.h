// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SAMPLER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SAMPLER_H_INCLUDED__

#include "nbl/asset/ISampler.h"

#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{

class NBL_API IGPUSampler : public asset::ISampler, public IBackendObject
{
    protected:
        virtual ~IGPUSampler() = default;

    public:
        IGPUSampler(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SParams& params) : ISampler(params), IBackendObject(std::move(dev)) {}

        // OpenGL: const GLuint* handle
        // Vulkan: const VkSampler
        virtual const void* getNativeHandle() const = 0;
};

}

#endif