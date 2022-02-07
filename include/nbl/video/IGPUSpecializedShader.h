// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/ISpecializedShader.h"

namespace nbl
{
namespace video
{
//! GPU Version of Specialized Shader
/*
	@see ISpecializedShader
*/

class IGPUSpecializedShader : public asset::ISpecializedShader
{
public:
    IGPUSpecializedShader(asset::ISpecializedShader::E_SHADER_STAGE _stage)
        : m_stage(_stage) {}

    asset::ISpecializedShader::E_SHADER_STAGE getStage() const { return m_stage; }

protected:
    virtual ~IGPUSpecializedShader() = default;

    const asset::ISpecializedShader::E_SHADER_STAGE m_stage;
};

}
}

#endif
