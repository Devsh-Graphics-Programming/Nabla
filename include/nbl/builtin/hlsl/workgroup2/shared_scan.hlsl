// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup2
{

template<uint32_t WorkgroupSizeLog2, uint32_t _SubgroupSizeLog2, uint32_t _ItemsPerInvocation>
struct Configuration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = uint16_t(_SubgroupSizeLog2);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;
    static_assert(WorkgroupSizeLog2>=_SubgroupSizeLog2, "WorkgroupSize cannot be smaller than SubgroupSize");

    NBL_CONSTEXPR_STATIC_INLINE uint16_t LevelCount = conditional_value<WorkgroupSize <= 4*SubgroupSize,uint16_t,1,2>::value;

    // must have at least enough level 0 outputs to feed a single subgroup
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupsPerVirtualWorkgroupLog2 = mpl::max<uint32_t, WorkgroupSizeLog2, 2*SubgroupSizeLog2>::value - SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t VirtualWorkgroupSize = uint32_t(0x1u) << (SubgroupsPerVirtualWorkgroupLog2 + SubgroupSizeLog2);
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t2 ItemsPerInvocation;    TODO? doesn't allow inline definitions for uint32_t2 for some reason, uint32_t[2] as well ; declaring out of line results in not constant expression
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_0 = conditional_value<LevelCount==1,uint32_t,uint32_t(0x1u)<<(WorkgroupSizeLog2-SubgroupSizeLog2),_ItemsPerInvocation>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_1 = uint32_t(0x1u) << (SubgroupsPerVirtualWorkgroupLog2 - SubgroupSizeLog2);
    static_assert(ItemsPerInvocation_1<=4, "3 level scan would have been needed with this config!");
};

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
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;

        subgroup2::reduction<params_t> reduction;
        if (glsl::gl_SubgroupID() == 0)
        {
            vector_t value = reduction(dataAccessor.get(glsl::gl_WorkGroupID().x * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex()));
            dataAccessor.set(glsl::gl_WorkGroupID().x * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex(), value);   // can be safely merged with top line?
        }
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

        if (glsl::gl_SubgroupID() == 0)
        {
            vector_t value;
            if (Exclusive)
            {
                subgroup2::exclusive_scan<params_t> excl_scan;
                value = excl_scan(dataAccessor.get(glsl::gl_WorkGroupID().x * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex()));
            }
            else
            {
                subgroup2::inclusive_scan<params_t> incl_scan;
                value = incl_scan(dataAccessor.get(glsl::gl_WorkGroupID().x * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex()));
            }
            dataAccessor.set(glsl::gl_WorkGroupID().x * Config::WorkgroupSize + workgroup::SubgroupContiguousIndex(), value);   // can be safely merged with above lines?
        }
    }
};

// 2-level scans
template<class Config, class BinOp, class device_capabilities>
struct reduce<Config, BinOp, 2, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;   // scratch smem accessor needs to be this type

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        BinOp binop;

        vector_lv0_t scan_local[Config::VirtualWorkgroupSize / Config::WorkgroupSize];
        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        // level 0 scan
        subgroup2::inclusive_scan<params_lv0_t> inclusiveScan0;
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            scan_local[idx] = inclusiveScan0(dataAccessor.get(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex));
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                scratchAccessor.set(virtualSubgroupID, scan_local[idx][Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        if (glsl::gl_SubgroupID() == 0)
        {
            scratchAccessor.set(invocationIndex, inclusiveScan1(scratchAccessor.get(invocationIndex)));
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // set as last element in scan (reduction)
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            dataAccessor.set(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, scratchAccessor.get(Config::SubgroupSize-1));
        }
    }
};

template<class Config, class BinOp, bool Exclusive, class device_capabilities>
struct scan<Config, BinOp, Exclusive, 2, device_capabilities>
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;   // scratch smem accessor needs to be this type

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        BinOp binop;

        vector_lv0_t scan_local[Config::VirtualWorkgroupSize / Config::WorkgroupSize];
        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        subgroup2::inclusive_scan<params_lv0_t> inclusiveScan0;
        // level 0 scan
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            scan_local[idx] = inclusiveScan0(dataAccessor.get(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex));
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                scratchAccessor.set(virtualSubgroupID, scan_local[idx][Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        if (glsl::gl_SubgroupID() == 0)
        {
            const vector_lv1_t shiftedInput = hlsl::mix(hlsl::promote<vector_lv1_t>(BinOp::identity), scratchAccessor.get(invocationIndex-1), bool(invocationIndex));
            scratchAccessor.set(invocationIndex, inclusiveScan1(shiftedInput));
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // combine with level 0
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
            const vector_lv1_t left = scratchAccessor.get(virtualSubgroupID);
            if (Exclusive)
            {
                scalar_t left_last_elem = hlsl::mix(BinOp::identity, glsl::subgroupShuffleUp<scalar_t>(scan_local[idx][Config::ItemsPerInvocation_0-1],1), bool(glsl::gl_SubgroupInvocationID()));
                [unroll]
                for (uint32_t i = 0; i < Config::ItemsPerInvocation_0; i++)
                    scan_local[idx][Config::ItemsPerInvocation_0-i-1] = binop(left, hlsl::mix(scan_local[idx][Config::ItemsPerInvocation_0-i-2], left_last_elem, (Config::ItemsPerInvocation_0-i-1==0)));
            }
            else
            {
                [unroll]
                for (uint32_t i = 0; i < Config::ItemsPerInvocation_0; i++)
                    scan_local[idx][i] = binop(left, scan_local[idx][i]);
            }
            dataAccessor.set(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local[idx]);
        }
    }
};

}

}
}
}

#endif
