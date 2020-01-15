#ifndef _IRR_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_
#define _IRR_EXT_FULL_SCREEN_TRIANGLE_FULL_SCREEN_TRIANGLE_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace FullScreenTriangle
{


inline auto createFullScreenTriangle(video::IVideoDriver* driver)
{
	std::tuple<core::smart_refctd_ptr<video::IGPUSpecializedShader>,asset::SVertexInputParams,asset::SPrimitiveAssemblyParams> retval;

	const char* source = R"===(
#version 430 core
const vec2 pos[3] = vec2[3](vec2(-1.0, 1.0),vec2(-1.0,-3.0),vec2( 3.0, 1.0));
const vec2 tc[3] = vec2[3](vec2( 0.0, 0.0),vec2( 0.0, 2.0),vec2( 2.0, 0.0));

layout(location = 0) out vec2 TexCoord;

void main()
{
    gl_Position = vec4(pos[gl_VertexIndex],0.0,1.0);
    TexCoord = tc[gl_VertexIndex];
}
	)===";
	auto shader = driver->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(source));
    video::IGPUSpecializedShader::SInfo specInfo({}, nullptr, "main", video::IGPUSpecializedShader::ESS_VERTEX);
	std::get<0>(retval) = driver->createGPUSpecializedShader(shader.get(), std::move(specInfo));

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

}
}
}

#endif

