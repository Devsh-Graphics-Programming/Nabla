// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/scan/virtual_workgroup.hlsl"

// ITEMS_PER_WG = WORKGROUP_SIZE
static const uint32_t SharedScratchSz = nbl::hlsl::workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;

// TODO: Can we make it a static variable?
groupshared uint32_t wgScratch[SharedScratchSz];

#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

template<uint16_t offset>
struct WGScratchProxy
{
    uint32_t get(const uint32_t ix)
    {
        return wgScratch[ix+offset];
    }
    void set(const uint32_t ix, const uint32_t value)
    {
        wgScratch[ix+offset] = value;
    }

    uint32_t atomicAdd(uint32_t ix, uint32_t val)
    {
        return nbl::hlsl::glsl::atomicAdd(wgScratch[ix + offset], val);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};
static WGScratchProxy<0> accessor;

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

struct ScanPushConstants
{
    nbl::hlsl::scan::Parameters_t scanParams;
    nbl::hlsl::scan::DefaultSchedulerParameters_t schedulerParams;
};

[[vk::push_constant]]
ScanPushConstants spc;

/**
 * Required since we rely on SubgroupContiguousIndex instead of 
 * gl_LocalInvocationIndex which means to match the global index 
 * we can't use the gl_GlobalInvocationID but an index based on 
 * SubgroupContiguousIndex.
 */
uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

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
    nbl::hlsl::scan::main<BINOP<Storage_t>, Storage_t, IS_SCAN, IS_EXCLUSIVE, uint16_t(WORKGROUP_SIZE), WGScratchProxy<0> >(accessor);
}