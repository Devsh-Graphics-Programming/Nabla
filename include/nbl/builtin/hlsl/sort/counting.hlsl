// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

#include "nbl/builtin/hlsl/sort/common.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

#ifndef _NBL_BUILTIN_HLSL_SORT_COUNTING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SORT_COUNTING_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace sort
{

NBL_CONSTEXPR uint32_t BucketsPerThread = ceil((float) BucketCount / WorkgroupSize);

groupshared uint32_t prefixScratch[BucketCount];

struct ScratchProxy
{
    uint32_t get(const uint32_t ix)
    {
        return prefixScratch[ix];
    }
    void set(const uint32_t ix, const uint32_t value)
    {
        prefixScratch[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

static ScratchProxy arithmeticAccessor;

groupshared uint32_t sdata[BucketCount];

template<typename KeyAccessor, typename ValueAccessor, typename ScratchAccessor>
struct counting
{
    void init(
        const CountingPushData data
    ) {
        in_key_addr         = data.inputKeyAddress;
        out_key_addr        = data.outputKeyAddress;
        in_value_addr       = data.inputValueAddress;
        out_value_addr      = data.outputValueAddress;
        scratch_addr        = data.scratchAddress;
        data_element_count  = data.dataElementCount;
        minimum             = data.minimum;
        elements_per_wt     = data.elementsPerWT;
    }

    void histogram()
    {
        uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++)
            sdata[BucketsPerThread * tid + i] = 0;
        uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize) * elements_per_wt;

        nbl::hlsl::glsl::barrier();

        for (int i = 0; i < elements_per_wt; i++)
        {
            int j = index + i * WorkgroupSize + tid;
            if (j >= data_element_count)
                break;
            uint32_t value = ValueAccessor(in_value_addr + sizeof(uint32_t) * j).template deref<4>().load();
            nbl::hlsl::glsl::atomicAdd(sdata[value - minimum], (uint32_t) 1);
        }

        nbl::hlsl::glsl::barrier();

        uint32_t sum = 0;
        uint32_t scan_sum = 0;

        for (int i = 0; i < BucketsPerThread; i++)
        {
            sum = nbl::hlsl::workgroup::exclusive_scan < nbl::hlsl::plus < uint32_t >, WorkgroupSize > ::
            template __call <ScratchProxy>
            (sdata[WorkgroupSize * i + tid], arithmeticAccessor);

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

            ScratchAccessor(scratch_addr + sizeof(uint32_t) * (WorkgroupSize * i + tid)).template deref<4>().atomicAdd(sum);
            if ((tid == WorkgroupSize - 1) && i > 0)
                ScratchAccessor(scratch_addr + sizeof(uint32_t) * (WorkgroupSize * i)).template deref<4>().atomicAdd(scan_sum);

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

            if ((tid == WorkgroupSize - 1) && i < (BucketsPerThread - 1))
            {
                scan_sum = sum + sdata[WorkgroupSize * i + tid];
                sdata[WorkgroupSize * (i + 1)] += scan_sum;
            }
        }
    }
                
    void scatter()
    {
        uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++)
            sdata[BucketsPerThread * tid + i] = 0;
        uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize) * elements_per_wt;

        nbl::hlsl::glsl::barrier();

        [unroll]
        for (int i = 0; i < elements_per_wt; i++)
        {
            int j = index + i * WorkgroupSize + tid;
            if (j >= data_element_count)
                break;
            uint32_t key = KeyAccessor(in_key_addr + sizeof(uint32_t) * j).template deref<4>().load();
            uint32_t value = ValueAccessor(in_value_addr + sizeof(uint32_t) * j).template deref<4>().load();
            nbl::hlsl::glsl::atomicAdd(sdata[value - minimum], (uint32_t) 1);
        }

        nbl::hlsl::glsl::barrier();

        [unroll]
        for (int i = 0; i < elements_per_wt; i++)
        {
            int j = index + i * WorkgroupSize + tid;
            if (j >= data_element_count)
                break;
            uint32_t key = KeyAccessor(in_key_addr + sizeof(uint32_t) * j).template deref<4>().load();
            uint32_t value = ValueAccessor(in_value_addr + sizeof(uint32_t) * j).template deref<4>().load();
            sdata[value - minimum] = ScratchAccessor(scratch_addr + sizeof(uint32_t) * (value - minimum)).template deref<4>().atomicAdd(1);
            KeyAccessor(out_key_addr + sizeof(uint32_t) * sdata[value - minimum]).template deref<4>().store(key);
            ValueAccessor(out_value_addr + sizeof(uint32_t) * sdata[value - minimum]).template deref<4>().store(value);
        }
    }

    uint64_t in_key_addr, out_key_addr;
    uint64_t in_value_addr, out_value_addr;
    uint64_t scratch_addr;
    uint32_t data_element_count;
    uint32_t minimum;
    uint32_t elements_per_wt;
};

}
}
}

#endif