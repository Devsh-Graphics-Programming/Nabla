// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_MESH_BUFFER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_MESH_BUFFER_H_INCLUDED__

#include <algorithm>

#include "nbl/asset/asset.h"
#include "IGPUBuffer.h"
#include "IGPUDescriptorSet.h"
#include "IGPURenderpassIndependentPipeline.h"

namespace nbl::video
{

class IGPUMeshBuffer final : public asset::IMeshBuffer<IGPUBuffer,IGPUDescriptorSet,IGPURenderpassIndependentPipeline>
{
        using base_t = asset::IMeshBuffer<IGPUBuffer,IGPUDescriptorSet,IGPURenderpassIndependentPipeline>;

    public:
        using base_t::base_t;
};

} // end namespace nbl::video

#endif


