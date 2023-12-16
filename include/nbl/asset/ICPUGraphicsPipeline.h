// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED_

#include "nbl/asset/IGraphicsPipeline.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/ICPURenderpass.h"

namespace nbl::asset
{

class ICPUGraphicsPipeline final : public IAsset, public IGraphicsPipeline<ICPURenderpassIndependentPipeline,ICPURenderpass>
{
        using base_t = IGraphicsPipeline<ICPURenderpassIndependentPipeline,ICPURenderpass>;

    public:
        using base_t::base_t;

        SCreationParams& getCreationParameters()
        {
            assert(!isImmutable_debug());
            return m_params;
        }

    protected:
        ~ICPUGraphicsPipeline() {}
};

}

#endif