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

namespace impl
{
template<uint16_t WorkgroupSizeLog2, uint16_t SubgroupSizeLog2>
struct virtual_wg_size_log2
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t levels = conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2),uint16_t,conditional_value<(WorkgroupSizeLog2>SubgroupSizeLog2*2+2),uint16_t,3,2>::value,1>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = mpl::max_v<uint32_t, WorkgroupSizeLog2-SubgroupSizeLog2, SubgroupSizeLog2>+SubgroupSizeLog2;
};

template<class VirtualWorkgroup, uint16_t BaseItemsPerInvocation, uint16_t WorkgroupSizeLog2, uint16_t SubgroupSizeLog2>
struct items_per_invocation
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ItemsPerInvocationProductLog2 = mpl::max_v<int16_t,WorkgroupSizeLog2-SubgroupSizeLog2*VirtualWorkgroup::levels,0>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value0 = BaseItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value1 = uint16_t(0x1u) << conditional_value<VirtualWorkgroup::levels==3, uint16_t,mpl::min_v<uint16_t,ItemsPerInvocationProductLog2,2>, ItemsPerInvocationProductLog2>::value;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value2 = uint16_t(0x1u) << mpl::max_v<int16_t,ItemsPerInvocationProductLog2-2,0>;
};
}

template<uint32_t WorkgroupSizeLog2, uint32_t _SubgroupSizeLog2, uint32_t _ItemsPerInvocation>
struct Configuration
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = uint16_t(_SubgroupSizeLog2);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;
    static_assert(WorkgroupSizeLog2>=_SubgroupSizeLog2, "WorkgroupSize cannot be smaller than SubgroupSize");

    // must have at least enough level 0 outputs to feed a single subgroup
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupsPerVirtualWorkgroupLog2 = mpl::max_v<uint32_t, WorkgroupSizeLog2-SubgroupSizeLog2, SubgroupSizeLog2>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupsPerVirtualWorkgroup = 0x1u << SubgroupsPerVirtualWorkgroupLog2;

    using virtual_wg_t = impl::virtual_wg_size_log2<WorkgroupSizeLog2, SubgroupSizeLog2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LevelCount = virtual_wg_t::levels;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t VirtualWorkgroupSize = uint16_t(0x1u) << virtual_wg_t::value;
    using items_per_invoc_t = impl::items_per_invocation<virtual_wg_t, _ItemsPerInvocation, WorkgroupSizeLog2, SubgroupSizeLog2>;
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t2 ItemsPerInvocation;    TODO? doesn't allow inline definitions for uint32_t2 for some reason, uint32_t[2] as well ; declaring out of line results in not constant expression
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_0 = items_per_invoc_t::value0;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_1 = items_per_invoc_t::value1;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_2 = items_per_invoc_t::value2;
    static_assert(ItemsPerInvocation_1<=4, "3 level scan would have been needed with this config!");

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemSize = conditional_value<LevelCount==3,uint32_t,SubgroupSize*ItemsPerInvocation_2,0>::value + SubgroupsPerVirtualWorkgroup*ItemsPerInvocation_1;
};

// special case when workgroup size 2048 and subgroup size 16 needs 3 levels and virtual workgroup size 4096 to get a full subgroup scan each on level 1 and 2 16x16x16=4096
// specializing with macros because of DXC bug: https://github.com/microsoft/DirectXShaderCom0piler/issues/7007
#define SPECIALIZE_CONFIG_CASE_2048_16(ITEMS_PER_INVOC) template<>\
struct Configuration<11, 4, ITEMS_PER_INVOC>\
{\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(0x1u) << 11u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = uint16_t(4u);\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = uint16_t(0x1u) << SubgroupSizeLog2;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupsPerVirtualWorkgroupLog2 = 7u;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupsPerVirtualWorkgroup = 128u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LevelCount = 3u;\
    NBL_CONSTEXPR_STATIC_INLINE uint16_t VirtualWorkgroupSize = uint16_t(0x1u) << 4096;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_0 = ITEMS_PER_INVOC;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_1 = 1u;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation_2 = 1u;\
};\

SPECIALIZE_CONFIG_CASE_2048_16(1)
SPECIALIZE_CONFIG_CASE_2048_16(2)
SPECIALIZE_CONFIG_CASE_2048_16(4)

#undef SPECIALIZE_CONFIG_CASE_2048_16


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
        vector_t value;
        dataAccessor.get(glsl::gl_WorkGroupID().x * Config::SubgroupSize + workgroup::SubgroupContiguousIndex(), value);
        value = reduction(value);
        dataAccessor.set(glsl::gl_WorkGroupID().x * Config::SubgroupSize + workgroup::SubgroupContiguousIndex(), value);   // can be safely merged with top line?
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
        dataAccessor.get(glsl::gl_WorkGroupID().x * Config::SubgroupSize + workgroup::SubgroupContiguousIndex(), value);
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
        dataAccessor.set(glsl::gl_WorkGroupID().x * Config::SubgroupSize + workgroup::SubgroupContiguousIndex(), value);   // can be safely merged with above lines?
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
    void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        using config_t = subgroup2::Configuration<Config::SubgroupSizeLog2>;
        using params_lv0_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_0, device_capabilities>;
        using params_lv1_t = subgroup2::ArithmeticParams<config_t, BinOp, Config::ItemsPerInvocation_1, device_capabilities>;
        BinOp binop;

        vector_lv0_t scan_local[Config::VirtualWorkgroupSize / Config::WorkgroupSize];
        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        // level 0 scan
        subgroup2::reduction<params_lv0_t> reduction0;
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            dataAccessor.get(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local[idx]);
            scan_local[idx] = reduction0(scan_local[idx]);
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                const uint32_t bankedIndex = (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.set(bankedIndex, scan_local[idx][Config::ItemsPerInvocation_0-1]);    // set last element of subgroup scan (reduction) to level 1 scan
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
                scratchAccessor.get(i*Config::SubgroupsPerVirtualWorkgroup+invocationIndex,lv1_val[i]);
            lv1_val = reduction1(lv1_val);
            scratchAccessor.set(invocationIndex, lv1_val[Config::ItemsPerInvocation_1-1]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // set as last element in scan (reduction)
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            scalar_t reduce_val;
            scratchAccessor.get(glsl::gl_SubgroupInvocationID(),reduce_val);
            dataAccessor.set(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, reduce_val);
        }
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

        vector_lv0_t scan_local[Config::VirtualWorkgroupSize / Config::WorkgroupSize];
        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        subgroup2::inclusive_scan<params_lv0_t> inclusiveScan0;
        // level 0 scan
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            dataAccessor.get(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local[idx]);
            scan_local[idx] = inclusiveScan0(scan_local[idx]);
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                const uint32_t bankedIndex = (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.set(bankedIndex, scan_local[idx][Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
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
                scratchAccessor.get(i*Config::SubgroupsPerVirtualWorkgroup+prevIndex,lv1_val[i]);
            vector_lv1_t shiftedInput = hlsl::mix(hlsl::promote<vector_lv1_t>(BinOp::identity), lv1_val, bool(invocationIndex));
            shiftedInput = inclusiveScan1(shiftedInput);
            scratchAccessor.set(invocationIndex, shiftedInput[Config::ItemsPerInvocation_1-1]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // combine with level 0
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
            scalar_t left;
            scratchAccessor.get(virtualSubgroupID,left);
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

// 3-level scans
template<class Config, class BinOp, class device_capabilities>
struct reduce<Config, BinOp, 3, device_capabilities>
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

        vector_lv0_t scan_local[Config::VirtualWorkgroupSize / Config::WorkgroupSize];
        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        // level 0 scan
        subgroup2::reduction<params_lv0_t> reduction0;
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            dataAccessor.get(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local[idx]);
            scan_local[idx] = reduction0(scan_local[idx]);
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                const uint32_t bankedIndex = (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.set(bankedIndex, scan_local[idx][Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
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
                scratchAccessor.get(i*Config::SubgroupsPerVirtualWorkgroup+invocationIndex,lv1_val[i]);
            lv1_val = reduction1(lv1_val);
            if (subgroup::ElectLast())
            {
                const uint32_t bankedIndex = (invocationIndex & (Config::ItemsPerInvocation_2-1)) * Config::SubgroupsPerVirtualWorkgroup + (invocationIndex/Config::ItemsPerInvocation_2);
                scratchAccessor.set(bankedIndex, lv1_val[Config::ItemsPerInvocation_1-1]);
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
                scratchAccessor.get(i*Config::SubgroupsPerVirtualWorkgroup+invocationIndex,lv2_val[i]);
            lv2_val = reduction2(lv2_val);
            scratchAccessor.set(invocationIndex, lv2_val[Config::ItemsPerInvocation_2-1]);
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // set as last element in scan (reduction)
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            scalar_t reduce_val;
            scratchAccessor.get(glsl::gl_SubgroupInvocationID(),reduce_val);
            dataAccessor.set(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, reduce_val);
        }
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

        vector_lv0_t scan_local[Config::VirtualWorkgroupSize / Config::WorkgroupSize];
        const uint32_t invocationIndex = workgroup::SubgroupContiguousIndex();
        subgroup2::inclusive_scan<params_lv0_t> inclusiveScan0;
        // level 0 scan
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            dataAccessor.get(glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize + idx * Config::WorkgroupSize + virtualInvocationIndex, scan_local[idx]);
            scan_local[idx] = inclusiveScan0(scan_local[idx]);
            if (subgroup::ElectLast())
            {
                const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
                const uint32_t bankedIndex = (virtualSubgroupID & (Config::ItemsPerInvocation_1-1)) * Config::SubgroupsPerVirtualWorkgroup + (virtualSubgroupID/Config::ItemsPerInvocation_1);
                scratchAccessor.set(bankedIndex, scan_local[idx][Config::ItemsPerInvocation_0-1]);   // set last element of subgroup scan (reduction) to level 1 scan
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // level 1 scan
        const uint32_t lv1_smem_size = Config::SubgroupsPerVirtualWorkgroup*Config::ItemsPerInvocation_1;
        subgroup2::inclusive_scan<params_lv1_t> inclusiveScan1;
        if (glsl::gl_SubgroupID() < lv1_smem_size)
        {
            vector_lv1_t lv1_val;
            [unroll]
            for (uint32_t i = 0; i < Config::ItemsPerInvocation_1; i++)
                scratchAccessor.get(i*Config::SubgroupsPerVirtualWorkgroup+invocationIndex,lv1_val[i]);
            lv1_val = inclusiveScan1(lv1_val);
            if (subgroup::ElectLast())
            {
                const uint32_t bankedIndex = (glsl::gl_SubgroupID() & (Config::ItemsPerInvocation_2-1)) * Config::SubgroupSize + (glsl::gl_SubgroupID()/Config::ItemsPerInvocation_2);
                scratchAccessor.set(lv1_smem_size+bankedIndex, lv1_val[Config::ItemsPerInvocation_1-1]);
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
                scratchAccessor.get(lv1_smem_size+i*Config::SubgroupSize+prevIndex,lv2_val[i]);
            vector_lv2_t shiftedInput = hlsl::mix(hlsl::promote<vector_lv2_t>(BinOp::identity), lv2_val, bool(invocationIndex));
            shiftedInput = inclusiveScan2(shiftedInput);

            // combine with level 1, only last element of each
            [unroll]
            for (uint32_t i = 0; i < Config::SubgroupsPerVirtualWorkgroup; i++)
            {
                scalar_t last_val;
                scratchAccessor.get((Config::ItemsPerInvocation_1-1)*Config::SubgroupsPerVirtualWorkgroup+(Config::SubgroupsPerVirtualWorkgroup-1-i),last_val);
                scalar_t val = hlsl::mix(hlsl::promote<vector_lv2_t>(BinOp::identity), lv2_val, bool(i));
                val = binop(last_val, shiftedInput[Config::ItemsPerInvocation_2-1]);
                scratchAccessor.set((Config::ItemsPerInvocation_1-1)*Config::SubgroupsPerVirtualWorkgroup+(Config::SubgroupsPerVirtualWorkgroup-1-i), last_val);
            }
        }
        scratchAccessor.workgroupExecutionAndMemoryBarrier();

        // combine with level 0
        [unroll]
        for (uint32_t idx = 0, virtualInvocationIndex = invocationIndex; idx < Config::VirtualWorkgroupSize / Config::WorkgroupSize; idx++)
        {
            const uint32_t virtualSubgroupID = idx * (Config::WorkgroupSize >> Config::SubgroupSizeLog2) + glsl::gl_SubgroupID();
            const scalar_t left;
            scratchAccessor.get(virtualSubgroupID, left);
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
