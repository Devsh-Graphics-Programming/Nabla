// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/asset/ISpecializedShader.h"

#include <utility>


namespace nbl::asset
{
//! Interface class for compute pipelines
/*
	This pipeline takes in Vulkan commands through 
	command buffers and processes them for computational work.

	A compute pipeline consists of a single compute shader 
	stage and the pipeline layout. The compute shader stage is capable
	of doing massive parallel arbitrary computations. The pipeline layout 
	connects the compute pipeline to the descriptor using the layout bindings.
*/
template<typename SpecShaderType>
class IComputePipeline
{
    public:
		constexpr static inline size_t SHADER_STAGE_COUNT = 1u;

        const SpecShaderType* getShader() const { return m_shader.get(); }

    protected:
		IComputePipeline(core::smart_refctd_ptr<SpecShaderType>&& _cs) : m_shader(std::move(_cs))
		{
            assert(m_shader->getStage()==IShader::ESS_COMPUTE);
        }
		virtual ~IComputePipeline() = default;

		core::smart_refctd_ptr<SpecShaderType> m_shader;
};

}


#endif