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
        glsl::barrier();
    }
};

static ScratchProxy arithmeticAccessor;

groupshared uint32_t sdata[BucketCount];

template<typename Key, typename KeyAccessor, typename ValueAccessor, typename ScratchAccessor>
struct counting
{
    void histogram(NBL_REF_ARG(KeyAccessor) key, NBL_REF_ARG(ScratchAccessor) scratch, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++) {
            uint32_t prev_bucket_count = WorkgroupSize * i;
            sdata[prev_bucket_count + tid] = 0;
        }

        uint32_t index = (glsl::gl_WorkGroupID().x * WorkgroupSize) * data.elementsPerWT;

        glsl::barrier();

        for (int i = 0; i < data.elementsPerWT; i++)
        {
            uint32_t prev_element_count = WorkgroupSize * i;
            int j = index + prev_element_count + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t k = key.get(j);
            glsl::atomicAdd(sdata[k - data.minimum], (uint32_t) 1);
        }

        glsl::barrier();

        uint32_t sum = 0;
        uint32_t scan_sum = 0;

        for (int i = 0; i < BucketsPerThread; i++)
        {
            uint32_t prev_bucket_count = WorkgroupSize * i;
            sum = workgroup::exclusive_scan < plus < uint32_t >, WorkgroupSize > ::
            template __call <ScratchProxy>
            (sdata[WorkgroupSize * i + tid], arithmeticAccessor);

            scratch.atomicAdd(prev_bucket_count + tid, sum);
            if ((tid == WorkgroupSize - 1) && i > 0)
                scratch.atomicAdd(prev_bucket_count, scan_sum);

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

            if ((tid == WorkgroupSize - 1) && i < (BucketsPerThread - 1))
            {
                scan_sum = sum + sdata[prev_bucket_count + tid];
                sdata[prev_bucket_count + WorkgroupSize] += scan_sum;
            }
        }
    }
                
    void scatter(NBL_REF_ARG(KeyAccessor) key, NBL_REF_ARG(ValueAccessor) val, NBL_REF_ARG(ScratchAccessor) scratch, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++) {
            uint32_t prev_bucket_count = WorkgroupSize * i;
            sdata[prev_bucket_count + tid] = 0;
        }

        uint32_t index = (glsl::gl_WorkGroupID().x * WorkgroupSize) * data.elementsPerWT;

        glsl::barrier();

        [unroll]
        for (int i = 0; i < data.elementsPerWT; i++)
        {
            uint32_t prev_element_count = WorkgroupSize * i;
            int j = index + prev_element_count + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t k = key.get(j);
            uint32_t v = val.get(j);
            sdata[k - data.minimum] = scratch.atomicAdd(k - data.minimum, (uint32_t) 1);
            key.set(sdata[k - data.minimum], k);
            val.set(sdata[k - data.minimum], v);
        }
    }
};

}
}
}

#endif