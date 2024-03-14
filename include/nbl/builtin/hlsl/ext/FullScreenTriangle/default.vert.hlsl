// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
//#include <nbl/builtin/glsl/utils/surface_transform.glsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace FullScreenTriangle
{

const static float32_t2 pos[3] = {
    float32_t2(-1.0,-1.0),
    float32_t2(-1.0, 3.0),
    float32_t2( 3.0,-1.0)
};
const static float32_t2 tc[3] = {
    float32_t2(0.0,0.0),
    float32_t2(0.0,2.0),
    float32_t2(2.0,0.0)
};

/*
layout (push_constant) uniform pushConstants
{
	layout (offset = 0) uint swapchainTransform;
} u_pushConstants;
*/

VertexAttributes main()
{
    using namespace nbl::hlsl::glsl;

    VertexAttributes retval;
//    vec2 pos = nbl_glsl_surface_transform_applyToNDC(u_pushConstants.swapchainTransform, pos[gl_VertexIndex]);
    spirv::Position = vec4(pos[gl_VertexIndex()], 0.f, 1.f);
    retval.uv = tc[gl_VertexIndex()];
    return retval;
}

}
}
}
}