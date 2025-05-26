// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic_config.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup2
{

namespace impl
{

template<class Config, class BinOp, uint16_t LevelCount, class device_capabilities>
struct reduce;

template<class Config, class BinOp, bool Exclusive, uint16_t LevelCount, class device_capabilities>
struct scan;

// 1-level scans
template<class Config, class BinOp, class device_capabilities>
struct reduce<Config, BinOp, 1, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    // doesn't use scratch smem, need as param?

    template<class DataAccessor, class ScratchAccessor>
    scalar_t __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;

        subgroup2::reduction<params_t> reduction;
        vector_t value;
        dataAccessor.template get<vector_t>(workgroup::SubgroupContiguousIndex(), value);
        value = reduction(value);
        return value[0];
        // dataAccessor.template set<vector_t>(workgroup::SubgroupContiguousIndex(), value);
    }
};

template<class Config, class BinOp, bool Exclusive, class device_capabilities>
struct scan<Config, BinOp, Exclusive, 1, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    // doesn't use scratch smem, need as param?

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;

        vector_t value;
        dataAccessor.template get<vector_t>(workgroup::SubgroupContiguousIndex(), value);
        if (Exclusive)
        {
            subgroup2::exclusive_scan<params_t> excl_scan;
            value = excl_scan(value);
        }
        else
        {
            subgroup2::inclusive_scan<params_t> incl_scan;
            value = incl_scan(value);
        }
        dataAccessor.template set<vector_t>(workgroup::SubgroupContiguousIndex(), value);   // can be safely merged with above lines?
    }
};

// 2-level scans
template<class Config, class BinOp, class device_capabilities>
struct reduce<Config, BinOp, 2, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;

    template<class DataAccessor, class ScratchAccessor>
    scalar_t __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        BinOp binop;

        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        // level 0 scan
        subgroup2::reduction<params_lv0_t> reduction0;
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            vector_lv0_t scan_local;
            dataAccessor.template get<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local);
            scan_local = reduction0(scan_local);
            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
            {
                const uint32_t virtualSubgroupID = Config::virtualSubgroupID(glsl::gl_SubgroupID(), idx);
                const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(virtualSubgroupID, Config::ItemsPerInvocation_1);    // (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.template set<scalar_t>(bankedIndex, scan_local[Config::ItemsPerInvocation_0-1]);    // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        subgroup2::reduction<params_lv1_t> reduction1;
        if (glsl::gl_SubgroupID() == 0)
        {
            vector_lv1_t lv1_val;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template get<scalar_t>(i*Config::SubgroupSize+invocationIndex,lv1_val[i]);
            lv1_val = reduction1(lv1_val);

            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
                scratchAccessor.template set<scalar_t>(0, lv1_val[Config::ItemsPerInvocation_1-1]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        scalar_t reduce_val;
        scratchAccessor.template get<scalar_t>(0,reduce_val);
        return reduce_val;
    }
};

template<class Config, class BinOp, bool Exclusive, class device_capabilities>
struct scan<Config, BinOp, Exclusive, 2, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        BinOp binop;

        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        subgroup2::inclusive_scan<params_lv0_t> inclusiveScan0;
        // level 0 scan
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            vector_lv0_t value;
            dataAccessor.template get<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);
            value = inclusiveScan0(value);
            dataAccessor.template set<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);
            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
            {
                const uint32_t virtualSubgroupID = Config::virtualSubgroupID(glsl::gl_SubgroupID(), idx);
                const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(virtualSubgroupID, Config::ItemsPerInvocation_1);    // (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.template set<scalar_t>(bankedIndex, value[Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        if (glsl::gl_SubgroupID() == 0)
        {
            vector_lv1_t lv1_val;
            const uint32_t prevIndex = invocationIndex-1;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template get<scalar_t>(i*Config::SubgroupSize+prevIndex,lv1_val[i]);
            lv1_val[0] = hlsl::mix(BinOp::identity, lv1_val[0], bool(invocationIndex));
            lv1_val = inclusiveScan1(lv1_val);
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template set<scalar_t>(i*Config::SubgroupSize+invocationIndex,lv1_val[i]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // combine with level 0
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            vector_lv0_t value;
            dataAccessor.template get<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);

            const uint32_t virtualSubgroupID = Config::virtualSubgroupID(glsl::gl_SubgroupID(), idx);
            const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(virtualSubgroupID, Config::ItemsPerInvocation_1);
            scalar_t left;
            scratchAccessor.template get<scalar_t>(bankedIndex,left);
            if (Exclusive)
            {
                scalar_t left_last_elem = hlsl::mix(BinOp::identity, glsl::subgroupShuffleUp<scalar_t>(value[Config::ItemsPerInvocation_0-1],1), bool(glsl::gl_SubgroupInvocationID()));
                [unroll]
                for (uint32_t i = Config::ItemsPerInvocation_0-1; i > 0; i--)
                    value[i] = binop(left, value[i-1]);
                value[0] = binop(left, left_last_elem);
            }
            else
            {
                [unroll]
                for (uint32_t i = 0; i < Config::ItemsPerInvocation_0; i++)
                    value[i] = binop(left, value[i]);
            }
            dataAccessor.template set<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);
        }
    }
};

// 3-level scans
template<class Config, class BinOp, class device_capabilities>
struct reduce<Config, BinOp, 3, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;
    using vector_lv2_t = vector<scalar_t, Config::ItemsPerInvocation_2>;

    template<class DataAccessor, class ScratchAccessor>
    scalar_t __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        using params_lv2_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_2, device_capabilities>;
        BinOp binop;

        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        // level 0 scan
        subgroup2::reduction<params_lv0_t> reduction0;
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            vector_lv0_t scan_local;
            dataAccessor.template get<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local);
            scan_local = reduction0(scan_local);
            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
            {
                const uint32_t virtualSubgroupID = Config::virtualSubgroupID(glsl::gl_SubgroupID(), idx);
                const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(virtualSubgroupID, Config::ItemsPerInvocation_1);    // (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.template set<scalar_t>(bankedIndex, scan_local[Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        subgroup2::reduction<params_lv1_t> reduction1;
        if (glsl::gl_SubgroupID() < Config::SubgroupSizeLog2*Config::ItemsPerInvocation_1)
        {
            vector_lv1_t lv1_val;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template get<scalar_t>(i*Config::SubgroupSize+invocationIndex,lv1_val[i]);
            lv1_val = reduction1(lv1_val);
            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
            {
                const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(invocationIndex, Config::ItemsPerInvocation_2);    // (invocationIndex & (Config::ItemsPerInvocation_2-1)) * Config::SubgroupsPerVirtualWorkgroup + (invocationIndex/Config::ItemsPerInvocation_2);
                scratchAccessor.template set<scalar_t>(bankedIndex, lv1_val[Config::ItemsPerInvocation_1-1]);
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 2 scan
        subgroup2::reduction<params_lv2_t> reduction2;
        if (glsl::gl_SubgroupID() == 0)
        {
            vector_lv2_t lv2_val;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_2; i++)
                scratchAccessor.template get<scalar_t>(i*Config::SubgroupSize+invocationIndex,lv2_val[i]);
            lv2_val = reduction2(lv2_val);
            scratchAccessor.template set<scalar_t>(invocationIndex, lv2_val[Config::ItemsPerInvocation_2-1]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        scalar_t reduce_val;
        scratchAccessor.template get<scalar_t>(0,reduce_val);
        return reduce_val;
    }
};

template<class Config, class BinOp, bool Exclusive, class device_capabilities>
struct scan<Config, BinOp, Exclusive, 3, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;
    using vector_lv2_t = vector<scalar_t, Config::ItemsPerInvocation_2>;

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        using params_lv2_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_2, device_capabilities>;
        BinOp binop;

        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        subgroup2::inclusive_scan<params_lv0_t> inclusiveScan0;
        // level 0 scan
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            vector_lv0_t value;
            dataAccessor.template get<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);
            value = inclusiveScan0(value);
            dataAccessor.template set<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);
            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
            {
                const uint32_t virtualSubgroupID = Config::virtualSubgroupID(glsl::gl_SubgroupID(), idx);
                const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(virtualSubgroupID, Config::ItemsPerInvocation_1);
                scratchAccessor.template set<scalar_t>(bankedIndex, value[Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        const uint32_t lv1_smem_size = Config::SubgroupsSize*Config::ItemsPerInvocation_1;
        subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        if (glsl::gl_SubgroupID() < lv1_smem_size)
        {
            vector_lv1_t lv1_val;
            const uint32_t prevIndex = invocationIndex-1;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template get<scalar_t>(i*Config::SubgroupSize+prevIndex,lv1_val[i]);
            lv1_val[0] = hlsl::mix(BinOp::identity, lv1_val[0], bool(invocationIndex));
            lv1_val = inclusiveScan1(lv1_val);
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template set<scalar_t>(i*Config::SubgroupSize+invocationIndex,lv1_val[i]);
            if (glsl::gl_SubgroupInvocationID()==Config::SubgroupSize-1)
            {
                const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(glsl::gl_SubgroupID(), Config::ItemsPerInvocation_2);  // (glsl::gl_SubgroupID() & (Config::ItemsPerInvocation_2-1)) * Config::SubgroupSize + (glsl::gl_SubgroupID()/Config::ItemsPerInvocation_2);
                scratchAccessor.template set<scalar_t>(lv1_smem_size+bankedIndex, lv1_val[Config::ItemsPerInvocation_1-1]);
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 2 scan
        subgroup2::inclusive_scan<params_lv2_t> inclusiveScan2;
        if (glsl::gl_SubgroupID() == 0)
        {
            vector_lv2_t lv2_val;
            const uint32_t prevIndex = invocationIndex-1;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_2; i++)
                scratchAccessor.template get<scalar_t>(lv1_smem_size+i*Config::SubgroupSize+prevIndex,lv2_val[i]);
            lv2_val[0] = hlsl::mix(hlsl::promote<vector_lv2_t>(BinOp::identity), lv2_val[0], bool(invocationIndex));
            lv2_val = inclusiveScan2(lv2_val);
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_2; i++)
                scratchAccessor.template set<scalar_t>(lv1_smem_size+i*Config::SubgroupSize+invocationIndex,lv2_val[i]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // combine with level 1
        if (glsl::gl_SubgroupID() < lv1_smem_size)
        {
            vector_lv1_t lv1_val;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template get<scalar_t>(i*Config::SubgroupSize+invocationIndex,lv1_val[i]);

            scalar_t lv2_scan;
            const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(glsl::gl_SubgroupID(), Config::ItemsPerInvocation_2);  // (glsl::gl_SubgroupID() & (Config::ItemsPerInvocation_2-1)) * Config::SubgroupSize + (glsl::gl_SubgroupID()/Config::ItemsPerInvocation_2);
            scratchAccessor.template set<scalar_t>(lv1_smem_size+bankedIndex, lv2_scan);

            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.template set<scalar_t>(i*Config::SubgroupSize+invocationIndex, binop(lv1_val[i],lv2_scan));
        }

        // combine with level 0
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            vector_lv0_t value;
            dataAccessor.template get<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);

            const uint32_t virtualSubgroupID = Config::virtualSubgroupID(glsl::gl_SubgroupID(), idx);
            const uint32_t bankedIndex = Config::sharedMemCoalescedIndex(virtualSubgroupID, Config::ItemsPerInvocation_1);
            scalar_t left;
            scratchAccessor.template get<scalar_t>(bankedIndex,left);
            if (Exclusive)
            {
                scalar_t left_last_elem = hlsl::mix(BinOp::identity, glsl::subgroupShuffleUp<scalar_t>(value[Config::ItemsPerInvocation_0-1],1), bool(glsl::gl_SubgroupInvocationID()));
                [unroll]
                for (uint32_t i = Config::ItemsPerInvocation_0-1; i > 0; i--)
                    value[i] = binop(left, value[i-1]);
                value[0] = binop(left, left_last_elem);
            }
            else
            {
                [unroll]
                for (uint32_t i = 0; i < Config::ItemsPerInvocation_0; i++)
                    value[i] = binop(left, value[i]);
            }
            dataAccessor.template set<vector_lv0_t>(idx * Config::WorkgroupSize + virtualInvocationIndex, value);
        }
    }
};

}

}
}
}

#endif
