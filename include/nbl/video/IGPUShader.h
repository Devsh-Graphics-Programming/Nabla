// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SHADER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SHADER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/ISPIR_VProgram.h"

namespace nbl
{
namespace video
{
//! GPU Version of Unspecialized Shader
/*
	@see IReferenceCounted
*/

class IGPUShader : public asset::IShader
{
protected:
    virtual ~IGPUShader() = default;
};

}
}

#endif
