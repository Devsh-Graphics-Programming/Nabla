// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

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
    void histogram(const KeyAccessor in_key, const ScratchAccessor scratch, const CountingPushData data)
    {
        uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++)
            sdata[BucketsPerThread * tid + i] = 0;
        uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize) * data.elementsPerWT;

        nbl::hlsl::glsl::barrier();

        for (int i = 0; i < data.elementsPerWT; i++)
        {
            int j = index + i * WorkgroupSize + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t key = in_key.get(j);// ValueAccessor(in_value_addr + sizeof(uint32_t) * j).template deref<4>().load();
            nbl::hlsl::glsl::atomicAdd(sdata[key - data.minimum], (uint32_t) 1);
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

            scratch.atomicAdd(WorkgroupSize * i + tid, sum);
            if ((tid == WorkgroupSize - 1) && i > 0)
                scratch.atomicAdd(WorkgroupSize * i, scan_sum);

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

            if ((tid == WorkgroupSize - 1) && i < (BucketsPerThread - 1))
            {
                scan_sum = sum + sdata[WorkgroupSize * i + tid];
                sdata[WorkgroupSize * (i + 1)] += scan_sum;
            }
        }
    }
                
    void scatter(const KeyAccessor in_key, const ValueAccessor in_val, const ScratchAccessor scratch, const KeyAccessor out_key, const ValueAccessor out_val, const CountingPushData data)
    {
        uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++)
            sdata[BucketsPerThread * tid + i] = 0;
        uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize) * data.elementsPerWT;

        nbl::hlsl::glsl::barrier();

        [unroll]
        for (int i = 0; i < data.elementsPerWT; i++)
        {
            int j = index + i * WorkgroupSize + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t key = in_key.get(j);
            uint32_t value = in_val.get(j);
            sdata[key - data.minimum] = scratch.atomicAdd(key - data.minimum, 1);
            out_key.set(sdata[key - data.minimum], key);
            out_val.set(sdata[key - data.minimum], value);
        }
    }

    KeyAccessor in_key, out_key;
    ValueAccessor in_val, out_val;
    ScratchAccessor scratch;
    uint32_t dataElementCount;
    uint32_t minimum;
    uint32_t elementsPerWT;
};

}
}
}

#endif