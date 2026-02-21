// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Frustum/builtin/hlsl/common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::ext::frustum;
// Push constants
[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
PSInput frustum_vertex_single() 
{
    PSInput output;
    float32_t3 vertex = getNDCCubeVertex();

    output.position = math::linalg::promoted_mul(pc.spc.instance.transform, vertex);
    output.color = pc.spc.instance.color;

    return output;
}
  // Vertex shader - batch mode (instanced)
[shader("vertex")]
PSInput frustum_vertex_instances() 
{
    PSInput output;
    const float32_t3 vertex = getNDCCubeVertex();
    InstanceData instance = vk::BufferPointer<InstanceData>(pc.ipc.pInstanceBuffer + sizeof(InstanceData) * glsl::gl_InstanceIndex()).Get();

    output.position = math::linalg::promoted_mul(instance.transform, vertex);
    output.color = instance.color;

    return output;
}

[shader("pixel")]
float32_t4 frustum_fragment(PSInput input) : SV_TARGET 
{
    float32_t4 outColor = input.color;

    return outColor;
}