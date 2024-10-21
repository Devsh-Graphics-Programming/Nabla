// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include <nbl/builtin/hlsl/surface_transform.h>

using namespace ::nbl::hlsl;
using namespace ::nbl::hlsl::ext::FullScreenTriangle;

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

[[vk::constant_id(0)]] const uint32_t SwapchainTransform = 0;


SVertexAttributes main()
{
    using namespace ::nbl::hlsl::glsl;

    spirv::Position.xy = SurfaceTransform::applyToNDC((SurfaceTransform::FLAG_BITS)SwapchainTransform,pos[gl_VertexIndex()]);
    spirv::Position.z = 0.f;
    spirv::Position.w = 1.f;

    SVertexAttributes retval;
    retval.uv = tc[gl_VertexIndex()];
    return retval;
}