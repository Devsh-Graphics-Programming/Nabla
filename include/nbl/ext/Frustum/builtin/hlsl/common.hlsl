// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FRUSTUM_BUILTIN_HLSL_COMMON_INCLUDED_
#define _NBL_EXT_FRUSTUM_BUILTIN_HLSL_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#endif

namespace nbl
{
namespace ext
{
namespace frustum
{

struct InstanceData
{
    hlsl::float32_t4x4 transform;
    hlsl::float32_t4 color;
};

struct SSinglePC
{
    InstanceData instance;
};
            
struct SInstancedPC
{
    uint64_t pInstanceBuffer;
};

struct PushConstants
{
    SSinglePC spc;
    SInstancedPC ipc;
};
#ifdef __HLSL_VERSION
struct PSInput
{
    float32_t4 position : SV_Position;
    nointerpolation float32_t4 color : TEXCOORD0;
};

float32_t3 getNDCCubeVertex()
{
    float32_t3 v = (hlsl::promote<uint32_t3>(hlsl::glsl::gl_VertexIndex()) >> uint32_t3(0,2,1)) & 0x1u;
    return v * float32_t3(2,2,1) + float32_t3(-1,-1,0);
}
#endif
          
}
}
}

#endif