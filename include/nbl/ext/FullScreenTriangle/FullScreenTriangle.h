// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_
#define _NBL_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_

#include "nabla.h"

namespace nbl
{
namespace ext
{
namespace FullScreenTriangle
{

inline auto createFullScreenTriangle(nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams)
{
	if (!cpu2gpuParams.assetManager)
		assert(false);

	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
	auto* assetManager = cpu2gpuParams.assetManager;

	std::tuple<core::smart_refctd_ptr<video::IGPUSpecializedShader>,asset::SVertexInputParams,asset::SPrimitiveAssemblyParams> retval;

	asset::IAsset::E_TYPE types[] = { asset::IAsset::ET_SPECIALIZED_SHADER,static_cast<asset::IAsset::E_TYPE>(0u) };
	auto found = assetManager->findAssets("nbl/builtin/specialized_shader/fullscreentriangle.vert", types);
	assert(found->size());
	auto contents = found->begin()->getContents();
	assert(!contents.empty());
	auto pShader = static_cast<asset::ICPUSpecializedShader*>((contents.begin()->get()));

	core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuSpecializedShader;
	{
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&pShader, &pShader + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			assert(false);

		gpuSpecializedShader = (*gpu_array)[0];
	}

	auto& inputParams = std::get<asset::SVertexInputParams>(retval);
	{
		inputParams.enabledBindingFlags = inputParams.enabledAttribFlags = 0u;
		for (size_t i=0ull; i<asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; i++)
			inputParams.attributes[i] = {0u,asset::EF_UNKNOWN,0u};
		for (size_t i=0ull; i<asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
			inputParams.bindings[i] = {0u,asset::EVIR_PER_VERTEX};
	}

	auto& assemblyParams = std::get<asset::SPrimitiveAssemblyParams>(retval);
	assemblyParams.primitiveRestartEnable = false;
	assemblyParams.primitiveType = asset::EPT_TRIANGLE_LIST;
	assemblyParams.tessPatchVertCount = 3u;

    return retval;
}

inline auto createFullScreenTriangle(nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams, core::smart_refctd_ptr<video::IGPUSpecializedShader>&& fragShader, core::smart_refctd_ptr<video::IGPUPipelineLayout>&& pipelineLayout, const asset::SBlendParams& blendParams={}, const asset::SRasterizationParams& rasterParams={})
{
	auto* logicalDevice = cpu2gpuParams.device;

	if (!logicalDevice)
		assert(false);

	auto protoPipeline = createFullScreenTriangle(cpu2gpuParams);

	video::IGPUSpecializedShader* shaders[] = {std::get<core::smart_refctd_ptr<video::IGPUSpecializedShader> >(protoPipeline).get(),fragShader.get()};
	auto pipeline = logicalDevice->createGPURenderpassIndependentPipeline(nullptr,std::move(pipelineLayout),shaders,shaders+2,std::get<asset::SVertexInputParams>(protoPipeline),blendParams,std::get<asset::SPrimitiveAssemblyParams>(protoPipeline),rasterParams);

	asset::SBufferBinding<video::IGPUBuffer> bindings[16];
	auto meshbuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{});
	meshbuffer->setIndexCount(3u);
	return meshbuffer;
}

}
}
}

#endif

