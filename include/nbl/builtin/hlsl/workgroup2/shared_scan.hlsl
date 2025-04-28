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

template<uint32_t _WorkgroupSize, uint32_t _SubgroupSizeLog2, uint32_t _ItemsPerInvocation>
struct Configuration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(_WorkgroupSize);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = uint16_t(_SubgroupSizeLog2);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;
    // NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocation = uint16_t(_ItemsPerInvocation);

    // must have at least enough level 0 outputs to feed a single subgroup
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupsPerVirtualWorkgroup = mpl::max<uint32_t, (WorkgroupSize >> SubgroupSizeLog2), SubgroupSize>::value; //TODO expression not constant apparently
    NBL_CONSTEXPR_STATIC_INLINE uint32_t VirtualWorkgroupSize = SubgroupsPerVirtualWorkgroup << SubgroupSizeLog2;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t2 ItemsPerInvocation;    TODO? doesn't allow inline definitions for uint32_t2 for some reason, uint32_t[2] as well ; declaring out of line results in not constant expression
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_0 = _ItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_1 = SubgroupsPerVirtualWorkgroup >> SubgroupSizeLog2;
    static_assert(ItemsPerInvocation_1<=4, "3 level scan would have been needed with this config!");
};

namespace impl
{

template<class Config, class BinOp, class device_capabilities>
struct reduce
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;   // scratch smem accessor needs to be this type

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)   // groupshared vector_lv1_t scratch[Config::SubgroupsPerVirtualWorkgroup]
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
            scan_local[idx] = inclusiveScan0(dataAccessor.get(idx * Config::WorkgroupSize + virtualInvocationIndex));
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                scratchAccessor.set(virtualSubgroupID, scan_local[idx][Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        // level 1 scan
        if (glsl::gl_SubgroupID() == 0)
        {
            scratchAccessor.set(invocationIndex, inclusiveScan1(scratchAccessor.get(invocationIndex)));
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // set as last element in scan (reduction)
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
            dataAccessor.set(idx * Config::WorkgroupSize + virtualInvocationIndex, scratchAccessor.get(Config::SubgroupsPerVirtualWorkgroup-1));
        }
    }
};

template<class Config, class BinOp, uint16_t ItemCount, bool Exclusive, class device_capabilities>
struct scan
{
    using scalar_t = typename BinOp::type_t;
    using vector_lv0_t = vector<scalar_t, Config::ItemsPerInvocation_0>;   // data accessor needs to be this type
    using vector_lv1_t = vector<scalar_t, Config::ItemsPerInvocation_1>;   // scratch smem accessor needs to be this type

    template<class DataAccessor, class ScratchAccessor>
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)   // groupshared vector_lv1_t scratch[Config::SubgroupsPerVirtualWorkgroup]
    {
        // // TODO get this working
        // // same thing for level 0
        // using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        // using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        // using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        // BinOp binop;

        // subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        // // level 1 scan
        // if (glsl::gl_SubgroupID() == 0)
        // {
        //     const vector_lv1_t shiftedInput = hlsl::mix(BinOp::identity, scratchAccessor.get(invocationIndex-1), bool(invocationIndex));
        //     scratchAccessor.set(invocationIndex, inclusiveScan1(shiftedInput));
        // }
        // scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // // combine with level 0
        // [unroll]
        // for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        // {
        //     const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
        //     dataAccessor.set(idx * Config::WorkgroupSize + virtualInvocationIndex, binop(scratchAccessor.get(virtualSubgroupID), scan_local[idx]));
        // }
    }
};

}

}
}
}

#endif
