// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_I_GPU_MESH_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_MESH_BUFFER_H_INCLUDED_

#include <algorithm>

#include "nbl/asset/asset.h"
#include "IGPUBuffer.h"
#include "IGPUDescriptorSet.h"
#include "IGPURenderpassIndependentPipeline.h"

namespace nbl::video
{

class NBL_API IGPUMeshBuffer final : public asset::IMeshBuffer<IGPUBuffer,IGPUDescriptorSet,IGPURenderpassIndependentPipeline>
{
    public:
        using base_t = asset::IMeshBuffer<IGPUBuffer,IGPUDescriptorSet,IGPURenderpassIndependentPipeline>;

        using base_t::base_t;
};

} // end namespace nbl::video

#endif


