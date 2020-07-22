#ifndef _IRR_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_
#define _IRR_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace FullScreenTriangle
{


inline auto createFullScreenTriangle(asset::IAssetManager* am, video::IVideoDriver* driver)
{
	std::tuple<core::smart_refctd_ptr<video::IGPUSpecializedShader>,asset::SVertexInputParams,asset::SPrimitiveAssemblyParams> retval;

	asset::IAsset::E_TYPE types[] = { asset::IAsset::ET_SPECIALIZED_SHADER,static_cast<asset::IAsset::E_TYPE>(0u) };
	auto found = am->findAssets("irr/builtin/specializedshaders/fullscreentriangle.vert",types);
	assert(found->size());
	auto first = found->begin();
	assert(!first->isEmpty());
	auto pShader = static_cast<asset::ICPUSpecializedShader*>((first->getContents().begin()->get()));
	std::get<core::smart_refctd_ptr<video::IGPUSpecializedShader> >(retval) = driver->getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(&pShader,&pShader+1u)->front();

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

inline auto createFullScreenTriangle(core::smart_refctd_ptr<video::IGPUSpecializedShader>&& fragShader, core::smart_refctd_ptr<video::IGPUPipelineLayout>&& pipelineLayout, asset::IAssetManager* am, video::IVideoDriver* driver, const asset::SBlendParams& blendParams={}, const asset::SRasterizationParams& rasterParams={})
{
	auto protoPipeline = createFullScreenTriangle(am,driver);

	video::IGPUSpecializedShader* shaders[] = {std::get<core::smart_refctd_ptr<video::IGPUSpecializedShader> >(protoPipeline).get(),fragShader.get()};
	auto pipeline = driver->createGPURenderpassIndependentPipeline(nullptr,std::move(pipelineLayout),shaders,shaders+2,std::get<asset::SVertexInputParams>(protoPipeline),blendParams,std::get<asset::SPrimitiveAssemblyParams>(protoPipeline),rasterParams);

	asset::SBufferBinding<video::IGPUBuffer> bindings[16];
	auto meshbuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{});
	meshbuffer->setIndexCount(3u);
	return meshbuffer;
}

}
}
}

#endif

