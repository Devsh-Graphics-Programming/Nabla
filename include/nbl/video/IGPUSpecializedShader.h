// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "nbl/asset/IShader.h" // only because of m_stage member
#include "nbl/asset/ISpecializedShader.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

//! GPU Version of Specialized Shader
/*
	@see ISpecializedShader
*/

class IGPUSpecializedShader : public asset::ISpecializedShader, public IBackendObject
{
	public:
		IGPUSpecializedShader(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)) {}

		virtual asset::IShader::E_SHADER_STAGE getStage() const = 0;

	protected:
		virtual ~IGPUSpecializedShader() = default;
};

}

#endif

