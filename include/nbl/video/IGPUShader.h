// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_SHADER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_SHADER_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IShader.h"

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{

//! GPU Version of Unspecialized Shader
/*
	@see IReferenceCounted
*/

class IGPUShader : public asset::IShader, public IBackendObject
{
    public:
        using SSpecInfo = asset::IShader::SSpecInfo<const IGPUShader>;

    protected:
        explicit IGPUShader(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IShader::E_SHADER_STAGE shaderStage, std::string&& filepathHint)
            : IBackendObject(std::move(dev)), IShader(shaderStage, std::move(filepathHint)) {}

        virtual ~IGPUShader() = default;
};

}

#endif
