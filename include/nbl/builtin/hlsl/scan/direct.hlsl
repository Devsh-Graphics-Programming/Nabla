// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/scan/declarations.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

struct ScanPushConstants
{
    nbl::hlsl::scan::Parameters_t scanParams;
    nbl::hlsl::scan::DefaultSchedulerParameters_t schedulerParams;
};

[[vk::push_constant]]
ScanPushConstants spc;

namespace nbl
{
namespace hlsl
{
namespace scan
{
Parameters_t getParameters()
{
    return spc.scanParams;
}

DefaultSchedulerParameters_t getSchedulerParameters()
{
    return spc.schedulerParams;
}

}
}
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    nbl::hlsl::scan::main();
}