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
    typename key_t = decltype(experimental::declval<KeyAccessor>().get(0)),
    bool robust=false
>
struct counting
{
    using this_t = counting<GroupSize, KeyBucketCount, KeyAccessor, ValueAccessor, HistogramAccessor, SharedAccessor, key_t>;

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

        // Parallel reads must be coalesced
        uint32_t index = workGroupIndex * GroupSize * params.elementsPerWT + tid;
        uint32_t endIndex = min(params.dataElementCount, index + GroupSize * params.elementsPerWT); // implicitly breaks when params.dataElementCount is reached

        for (; index < endIndex; index += GroupSize)
        {
            uint32_t k;
            key.get(index, k);
            if (robust && (k<params.minimum || k>params.maximum) )
                continue;
            sdata.atomicAdd(k - params.minimum, (uint32_t) 1);
        }
    }

    void histogram(
            NBL_REF_ARG( KeyAccessor) key,
            NBL_REF_ARG(HistogramAccessor) histogram,
            NBL_REF_ARG(SharedAccessor) sdata,
            const CountingParameters<key_t> params
    )
    {
        build_histogram(key, sdata, params);

        // wait for the histogramming to finish
        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t tid = workgroup::SubgroupContiguousIndex();
        // because first chunk of histogram and workgroup scan scratch are aliased
        uint32_t histogram_value;
        sdata.get(tid, histogram_value);

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t sum = inclusive_scan(histogram_value, sdata);
        histogram.atomicAdd(tid, sum);

        const bool is_last_wg_invocation = tid == (GroupSize-_static_cast<uint16_t>(1));
        const static uint16_t RoundedKeyBucketCount = (KeyBucketCount-_static_cast<uint16_t>(1))/GroupSize+_static_cast<uint16_t>(1);

        for (int i = 1; i < RoundedKeyBucketCount; i++)
        {
            uint32_t keyBucketStart = GroupSize * i;
            uint32_t vid = tid + keyBucketStart;

            // no if statement about the last iteration needed
            if (is_last_wg_invocation)
            {
                uint32_t beforeSum;
                sdata.get(keyBucketStart, beforeSum);
                sdata.set(keyBucketStart, beforeSum + sum);
            }

            // propagate last block tail to next block head and protect against subsequent scans stepping on each other's toes
            sdata.workgroupExecutionAndMemoryBarrier();

            // no aliasing anymore
            uint32_t atVid;
            sdata.get(vid, atVid);
            sum = inclusive_scan(atVid, sdata);
            if (vid < KeyBucketCount) {
                histogram.atomicAdd(vid, sum);
            }
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

        // wait for the histogramming to finish
        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t tid = workgroup::SubgroupContiguousIndex();
        const uint32_t shift = glsl::gl_SubgroupSize() * workGroupIndex;

        for (uint32_t vtid=tid; vtid<KeyBucketCount; vtid+=GroupSize)
        {
            // have to use modulo operator in case `KeyBucketCount<=2*GroupSize`, better hope KeyBucketCount is Power of Two
            const uint32_t shifted_tid = (vtid + shift) % KeyBucketCount;
            const uint32_t bucket_value;
            sdata.get(shifted_tid, bucket_value);
            const uint32_t firstOutputIndex = histogram.atomicSub(shifted_tid, bucket_value) - bucket_value;

            sdata.set(shifted_tid, firstOutputIndex);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        uint32_t index = workGroupIndex * GroupSize * params.elementsPerWT + tid;
        uint32_t endIndex = min(params.dataElementCount, index + GroupSize * params.elementsPerWT);

        [unroll]
        for (; index < endIndex; index += GroupSize)
        {
            key_t k;
            key.get(index, k);
            if (robust && (k<params.minimum || k>params.maximum) )
                continue;
            uint32_t v;
            val.get(index, v);
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