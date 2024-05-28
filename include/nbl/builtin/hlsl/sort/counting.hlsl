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

    uint32_t toroidal_histogram_add(uint32_t tid, uint32_t sum, NBL_REF_ARG(HistogramAccessor) histogram, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> params)
    {
        sdata.workgroupExecutionAndMemoryBarrier();

        sdata.set(tid % GroupSize, sum);
        uint32_t shifted_tid = (tid + glsl::gl_SubgroupSize() * params.workGroupIndex) % GroupSize;

        sdata.workgroupExecutionAndMemoryBarrier();

        return histogram.atomicAdd((tid / GroupSize) * GroupSize + shifted_tid, sdata.get(shifted_tid));
    }

    void build_histogram(NBL_REF_ARG( KeyAccessor) key, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> params)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        for (; tid < KeyBucketCount; tid += GroupSize) {
            sdata.set(tid, 0);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t index = params.workGroupIndex * GroupSize * params.elementsPerWT + tid % GroupSize;
        uint32_t endIndex = min(params.dataElementCount, index + GroupSize * params.elementsPerWT);

        for (; index < endIndex; index += GroupSize)
        {
            uint32_t k = key.get(index);
            if (robust && (k<params.minimum || k>params.maximum) )
                continue;
            sdata.atomicAdd(k - params.minimum, (uint32_t) 1);
        }

        sdata.workgroupExecutionAndMemoryBarrier();
    }

    void histogram(NBL_REF_ARG( KeyAccessor) key, NBL_REF_ARG(HistogramAccessor) histogram, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> params)
    {
        build_histogram(key, sdata, params);

        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t histogram_value = sdata.get(tid);

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t sum = inclusive_scan(histogram_value, sdata);
        toroidal_histogram_add(tid, sum, histogram, sdata, params);

        const bool is_last_wg_invocation = tid == (GroupSize - 1);
        const uint16_t adjusted_key_bucket_count = ((KeyBucketCount - 1) / GroupSize + 1) * GroupSize;

        for (tid += GroupSize; tid < adjusted_key_bucket_count; tid += GroupSize)
        {
            if (is_last_wg_invocation)
            {
                uint32_t startIndex = tid - tid % GroupSize;
                sdata.set(startIndex, sdata.get(startIndex) + sum);
            }

            sum = inclusive_scan(sdata.get(tid), sdata);
            toroidal_histogram_add(tid, sum, histogram, sdata, params);
        }
    }
                
    void scatter(NBL_REF_ARG(KeyAccessor) key, NBL_REF_ARG(ValueAccessor) val, NBL_REF_ARG(HistogramAccessor) histogram, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<Key> params)
    {
        build_histogram(key, sdata, params);

        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t shifted_tid = (tid + glsl::gl_SubgroupSize() * params.workGroupIndex) % GroupSize;

        for (; shifted_tid < KeyBucketCount; shifted_tid += GroupSize)
        {
            uint32_t bucket_value = sdata.get(shifted_tid);
            uint32_t exclusive_value = histogram.atomicSub(shifted_tid, bucket_value) - bucket_value;

            sdata.set(shifted_tid, exclusive_value);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t index = params.workGroupIndex * GroupSize * params.elementsPerWT + tid;
        uint32_t endIndex = min(params.dataElementCount, index + GroupSize * params.elementsPerWT);

        [unroll]
        for (; index < endIndex; index += GroupSize)
        {
            const Key k = key.get(index);
            if (robust && (k<params.minimum || k>params.maximum) )
                continue;
            const uint32_t v = val.get(index);
            const uint32_t sortedIx = sdata.atomicAdd(k - params.minimum, 1);
            key.set(sortedIx, k);
            val.set(sortedIx, v);
        }
    }
};

}
}
}

#endif