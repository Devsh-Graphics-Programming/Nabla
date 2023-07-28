// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_


#include "nbl/asset/IRenderpassIndependentPipeline.h"

#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/IGPUPipelineLayout.h"


namespace nbl::video
{

//! GPU Version of Renderpass Independent Pipeline
/*
	@see IRenderpassIndependentPipeline
*/
class IGPURenderpassIndependentPipeline : public IBackendObject, public asset::IRenderpassIndependentPipeline<IGPUSpecializedShader,IGPUPipelineLayout>
{
		using base_t = asset::IRenderpassIndependentPipeline<IGPUSpecializedShader,IGPUPipelineLayout>;

	public:
		IGPURenderpassIndependentPipeline(
			core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
			IGPUSpecializedShader* const* _shadersBegin, IGPUSpecializedShader* const* _shadersEnd,
			const asset::SVertexInputParams& _vertexInputParams,
			const asset::SBlendParams& _blendParams,
			const asset::SPrimitiveAssemblyParams& _primAsmParams,
			const asset::SRasterizationParams& _rasterParams
		) : IBackendObject(std::move(dev)), base_t(std::move(_layout),_shadersBegin,_shadersEnd,_vertexInputParams,_blendParams,_primAsmParams,_rasterParams) {}

		struct SCreationParams
		{
			core::smart_refctd_ptr<IGPUPipelineLayout> layout;
			core::smart_refctd_ptr<const IGPUSpecializedShader> shaders[GRAPHICS_SHADER_STAGE_COUNT];
			asset::SVertexInputParams vertexInput;
			asset::SBlendParams blend;
			asset::SPrimitiveAssemblyParams primitiveAssembly;
			asset::SRasterizationParams rasterization;
		};

	protected:
		virtual ~IGPURenderpassIndependentPipeline() = default;
};

}

#endif