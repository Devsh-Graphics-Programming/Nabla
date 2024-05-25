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

template<uint16_t GroupSize, uint16_t KeyBucketCount, typename Key, typename KeyAccessor, typename ValueAccessor, typename ScratchAccessor, typename SharedAccessor>
struct counting
{
    void histogram(NBL_REF_ARG( KeyAccessor) key, NBL_REF_ARG(ScratchAccessor) scratch, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t buckets_per_thread = (KeyBucketCount + GroupSize - 1) / GroupSize;

        [unroll]
        for (int i = 0; i < buckets_per_thread; i++) {
            uint32_t prev_bucket_count = GroupSize * i;
            sdata.set(prev_bucket_count + tid, 0);
        }

        uint32_t index = (glsl::gl_WorkGroupID().x * GroupSize) * data.elementsPerWT;

        sdata.workgroupExecutionAndMemoryBarrier();

        for (int i = 0; i < data.elementsPerWT; i++)
        {
            uint32_t prev_element_count = GroupSize * i;
            int j = index + prev_element_count + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t k = key.get(j);
            sdata.atomicAdd(k - data.minimum, (uint32_t) 1);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t sum = 0;
        uint32_t scan_sum = 0;

        for (int i = 0; i < buckets_per_thread; i++)
        {
            uint32_t prev_bucket_count = GroupSize * i;
            uint32_t histogram_value = sdata.get(prev_bucket_count + tid);
            sum = workgroup::exclusive_scan < plus < uint32_t >, GroupSize > ::
            template __call <SharedAccessor>
            (histogram_value, sdata);

            scratch.atomicAdd(prev_bucket_count + tid, sum);
            if ((tid == GroupSize - 1) && i > 0)
                scratch.atomicAdd(prev_bucket_count, scan_sum);

            sdata.set(prev_bucket_count + tid, histogram_value);

            sdata.workgroupExecutionAndMemoryBarrier();

            if ((tid == GroupSize - 1) && i < (buckets_per_thread - 1))
            {
                scan_sum = sum + sdata.get(prev_bucket_count + tid);
                sdata.atomicAdd(prev_bucket_count + GroupSize, scan_sum);
            }
        }
    }
                
    void scatter(NBL_REF_ARG(KeyAccessor) key, NBL_REF_ARG(ValueAccessor) val, NBL_REF_ARG(ScratchAccessor) scratch, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t buckets_per_thread = (KeyBucketCount + GroupSize - 1) / GroupSize;

        [unroll]
        for (int i = 0; i < buckets_per_thread; i++) {
            uint32_t prev_bucket_count = GroupSize * i;
            sdata.set(prev_bucket_count + tid, 0);
        }

        uint32_t index = (glsl::gl_WorkGroupID().x * GroupSize) * data.elementsPerWT;

        sdata.workgroupExecutionAndMemoryBarrier();

        [unroll]
        for (int i = 0; i < data.elementsPerWT; i++)
        {
            uint32_t prev_element_count = GroupSize * i;
            int j = index + prev_element_count + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t k = key.get(j);
            uint32_t v = val.get(j);
            sdata.set(k - data.minimum, scratch.atomicAdd(k - data.minimum, (uint32_t) 1));
            key.set(sdata.get(k - data.minimum), k);
            val.set(sdata.get(k - data.minimum), v);
        }
    }
};

}
}
}

#endif