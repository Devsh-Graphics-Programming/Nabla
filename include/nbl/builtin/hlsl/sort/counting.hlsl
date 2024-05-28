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

template<
    uint16_t GroupSize,
    uint16_t KeyBucketCount,
    typename Key,
    typename KeyAccessor,
    typename ValueAccessor,
    typename HistogramAccessor,
    typename SharedAccessor,
    bool robust=false
>
struct counting
{
    uint32_t inclusive_scan(uint32_t value, NBL_REF_ARG(SharedAccessor) sdata)
    {
        return workgroup::inclusive_scan < plus < uint32_t >, GroupSize >::
                template __call <SharedAccessor>(value, sdata);
    }

    void build_histogram(NBL_REF_ARG( KeyAccessor) key, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t buckets_per_thread = (KeyBucketCount + GroupSize - 1) / GroupSize;
        uint32_t baseIndex = (glsl::gl_WorkGroupID().x * GroupSize) * data.elementsPerWT;

        for (int i = 0; i < buckets_per_thread; i++) {
            uint32_t prev_bucket_count = GroupSize * i;
            sdata.set(prev_bucket_count + tid, 0);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        for (int i = 0; i < data.elementsPerWT; i++)
        {
            uint32_t prev_element_count = GroupSize * i;
            int j = baseIndex + prev_element_count + tid;
            if (j >= data.dataElementCount)
                break;
            uint32_t k = key.get(j);
            if (robust && (k<data.minimum || k>data.maximum) )
                continue;
            sdata.atomicAdd(k - data.minimum, (uint32_t) 1);
        }

        sdata.workgroupExecutionAndMemoryBarrier();
    }

    void histogram(NBL_REF_ARG( KeyAccessor) key, NBL_REF_ARG(HistogramAccessor) histogram, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t buckets_per_thread = (KeyBucketCount + GroupSize - 1) / GroupSize;

        build_histogram(key, sdata, data);

        uint32_t histogram_value = sdata.get(tid);

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t sum = inclusive_scan(histogram_value, sdata);
        histogram.atomicAdd(tid, sum);

        for (int i = 1; i < buckets_per_thread; i++)
        {
            uint32_t prev_bucket_count = GroupSize * i;

            if (tid == GroupSize - 1) {
                sdata.atomicAdd(prev_bucket_count, sum);
            }

            sdata.workgroupExecutionAndMemoryBarrier();

            uint32_t index = prev_bucket_count + tid;
            sum = inclusive_scan(sdata.get(index), sdata);

            histogram.atomicAdd(prev_bucket_count + tid, sum);
        }
    }
                
    void scatter(NBL_REF_ARG(KeyAccessor) key, NBL_REF_ARG(ValueAccessor) val, NBL_REF_ARG(HistogramAccessor) histogram, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> data)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t buckets_per_thread = (KeyBucketCount + GroupSize - 1) / GroupSize;

        build_histogram(key, sdata, data);

        for (int i = 0; i < buckets_per_thread; i++)
        {
            uint32_t prev_bucket_count = GroupSize * i;
            uint32_t index = prev_bucket_count + tid;
            uint32_t bucket_value = sdata.get(index);
            uint32_t exclusive_value = histogram.atomicSub(index, bucket_value) - bucket_value;

            sdata.set(index, exclusive_value);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t baseIndex = (glsl::gl_WorkGroupID().x * GroupSize) * data.elementsPerWT;

        [unroll]
        for (int i = 0; i < data.elementsPerWT; i++)
        {
            uint32_t prev_element_count = GroupSize * i;
            int j = baseIndex + prev_element_count + tid;
            if (j >= data.dataElementCount)
                break;
            const Key k = key.get(j);
            if (robust && (k<data.minimum || k>data.maximum) )
                continue;
            const uint32_t v = val.get(j);
            const uint32_t sortedIx = sdata.atomicAdd(k - data.minimum, 1);
            key.set(sortedIx, k);
            val.set(sortedIx, v);
        }
    }
};

}
}
}

#endif