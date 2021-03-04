// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SHADER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SHADER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/ISPIR_VProgram.h"
#include "nbl/video/IBackendObject.h"

namespace nbl
{
namespace video
{

//! GPU Version of Unspecialized Shader
/*
	@see IReferenceCounted
*/

class IGPUShader : public asset::IShader, public IBackendObject
{
    protected:
        explicit IGPUShader(ILogicalDevice* dev) : IBackendObject(dev) {}

        virtual ~IGPUShader() = default;
};

}
}

#endif
