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
    typename KeyAccessor,
    typename ValueAccessor,
    typename HistogramAccessor,
    typename SharedAccessor,
    bool robust=false
>
struct counting
{
    using key_t = decltype(impl::declval < KeyAccessor > ().get(0));
    using this_t = counting<GroupSize, KeyBucketCount, KeyAccessor, ValueAccessor, HistogramAccessor, SharedAccessor>;

    static this_t create(const uint32_t workGroupIndex)
    {
        this_t retval;
        retval.workGroupIndex = workGroupIndex;
        return retval;
    }

    uint32_t inclusive_scan(uint32_t value, NBL_REF_ARG(SharedAccessor) sdata)
    {
        return workgroup::inclusive_scan < plus < uint32_t >, GroupSize >::
                template __call <SharedAccessor>(value, sdata);
    }

    void build_histogram(NBL_REF_ARG( KeyAccessor) key, NBL_REF_ARG(SharedAccessor) sdata, const CountingParameters<key_t> params)
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        for (uint32_t vid = tid; vid < KeyBucketCount; vid += GroupSize) {
            sdata.set(vid, 0);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t index = workGroupIndex * GroupSize * params.elementsPerWT + tid;
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

    void histogram(
            NBL_REF_ARG( KeyAccessor) key,
            NBL_REF_ARG(HistogramAccessor) histogram,
            NBL_REF_ARG(SharedAccessor) sdata,
            const CountingParameters<key_t> params
    )
    {
        build_histogram(key, sdata, params);

        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t histogram_value = sdata.get(tid);

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t sum = inclusive_scan(histogram_value, sdata);
        histogram.atomicAdd(tid, sum);

        const bool is_last_wg_invocation = tid == (GroupSize - 1);
        const uint16_t adjusted_key_bucket_count = ((KeyBucketCount - 1) / GroupSize + 1) * GroupSize;

        for (uint32_t vid = tid + GroupSize; vid < adjusted_key_bucket_count; vid += GroupSize)
        {
            if (is_last_wg_invocation)
            {
                uint32_t startIndex = vid - tid;
                sdata.set(startIndex, sdata.get(startIndex) + sum);
            }

            sum = inclusive_scan(sdata.get(vid), sdata);
            histogram.atomicAdd(vid, sum);
        }
    }
                
    void scatter(
            NBL_REF_ARG( KeyAccessor) key,
            NBL_REF_ARG(ValueAccessor) val,
            NBL_REF_ARG(HistogramAccessor) histogram,
            NBL_REF_ARG(SharedAccessor) sdata,
            const CountingParameters<key_t> params
    )
    {
        build_histogram(key, sdata, params);

        uint32_t tid = workgroup::SubgroupContiguousIndex();
        uint32_t shifted_tid = (tid + glsl::gl_SubgroupSize() * workGroupIndex) % GroupSize;

        for (; shifted_tid < KeyBucketCount; shifted_tid += GroupSize)
        {
            uint32_t bucket_value = sdata.get(shifted_tid);
            uint32_t exclusive_value = histogram.atomicSub(shifted_tid, bucket_value) - bucket_value;

            sdata.set(shifted_tid, exclusive_value);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t index = workGroupIndex * GroupSize * params.elementsPerWT + tid;
        uint32_t endIndex = min(params.dataElementCount, index + GroupSize * params.elementsPerWT);

        [unroll]
        for (; index < endIndex; index += GroupSize)
        {
            const key_t k = key.get(index);
            if (robust && (k<params.minimum || k>params.maximum) )
                continue;
            const uint32_t v = val.get(index);
            const uint32_t sortedIx = sdata.atomicAdd(k - params.minimum, 1);
            key.set(sortedIx, k);
            val.set(sortedIx, v);
        }
    }

    uint32_t workGroupIndex;
};

}
}
}

#endif