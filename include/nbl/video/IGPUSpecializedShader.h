// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__


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
		IGPUSpecializedShader(core::smart_refctd_ptr<const ILogicalDevice>&& dev, asset::ISpecializedShader::E_SHADER_STAGE _stage) : IBackendObject(std::move(dev)), m_stage(_stage) {}

		asset::ISpecializedShader::E_SHADER_STAGE getStage() const { return m_stage; }

	protected:
		virtual ~IGPUSpecializedShader() = default;

		const asset::ISpecializedShader::E_SHADER_STAGE m_stage;
};

}

#endif

