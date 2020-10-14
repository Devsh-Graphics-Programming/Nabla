// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SHADER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ISPIR_VProgram.h"

namespace irr
{
namespace video
{

class IGPUShader : public asset::IShader
{
    protected:
        virtual ~IGPUShader() = default;
};

}
}

#endif
