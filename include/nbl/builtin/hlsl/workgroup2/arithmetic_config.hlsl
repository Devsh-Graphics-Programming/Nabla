// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_CONFIG_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_CONFIG_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tuple.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup2
{

namespace impl
{
template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2>
struct virtual_wg_size_log2
{
    #define DEFINE_ASSIGN(TYPE,ID,...) NBL_CONSTEXPR_STATIC_INLINE TYPE ID = __VA_ARGS__;
    #define MAX(TYPE,ARG1,ARG2) mpl::max_v<TYPE, ARG1, ARG2>
    #define SELECT(TYPE,COND,TRUE_VAL,FALSE_VAL) conditional_value<COND,TYPE,TRUE_VAL,FALSE_VAL>::value
    #include "impl/virtual_wg_size_def.hlsl"
    #undef SELECT
    #undef MAX
    #undef DEFINE_ASSIGN
    
    // must have at least enough level 0 outputs to feed a single subgroup
    static_assert(WorkgroupSizeLog2>=SubgroupSizeLog2, "WorkgroupSize cannot be smaller than SubgroupSize");
    static_assert(WorkgroupSizeLog2<=SubgroupSizeLog2*3+4, "WorkgroupSize cannot be larger than (SubgroupSize^3)*16");
};

template<class VirtualWorkgroup, uint16_t BaseItemsPerInvocation>
struct items_per_invocation
{
    #define DEFINE_ASSIGN(TYPE,ID,...) NBL_CONSTEXPR_STATIC_INLINE TYPE ID = __VA_ARGS__;
    #define VIRTUAL_WG_SIZE VirtualWorkgroup::
    #define MIN(TYPE,ARG1,ARG2) mpl::min_v<TYPE, ARG1, ARG2>
    #define MAX(TYPE,ARG1,ARG2) mpl::max_v<TYPE, ARG1, ARG2>
    #define SELECT(TYPE,COND,TRUE_VAL,FALSE_VAL) conditional_value<COND,TYPE,TRUE_VAL,FALSE_VAL>::value
    #include "impl/items_per_invoc_def.hlsl"
    #undef SELECT
    #undef MAX
    #undef MIN
    #undef VIRTUAL_WG_SIZE
    #undef DEFINE_ASSIGN

    using ItemsPerInvocation = tuple<integral_constant<uint16_t,value0>,integral_constant<uint16_t,value1>,integral_constant<uint16_t,value2> >;
};
}

template<uint16_t _WorkgroupSizeLog2, uint16_t _SubgroupSizeLog2, uint16_t _ItemsPerInvocation>
struct ArithmeticConfiguration
{
    using virtual_wg_t = impl::virtual_wg_size_log2<_WorkgroupSizeLog2, _SubgroupSizeLog2>;
    using items_per_invoc_t = impl::items_per_invocation<virtual_wg_t, _ItemsPerInvocation>;
    using ItemsPerInvocation = typename items_per_invoc_t::ItemsPerInvocation;

    #define DEFINE_ASSIGN(TYPE,ID,...) NBL_CONSTEXPR_STATIC_INLINE TYPE ID = __VA_ARGS__;
    #define VIRTUAL_WG_SIZE virtual_wg_t::
    #define ITEMS_PER_INVOC items_per_invoc_t::
    #define MAX(TYPE,ARG1,ARG2) mpl::max_v<TYPE, ARG1, ARG2>
    #define SELECT(TYPE,COND,TRUE_VAL,FALSE_VAL) conditional_value<COND,TYPE,TRUE_VAL,FALSE_VAL>::value
    #include "impl/arithmetic_config_def.hlsl"
    #undef SELECT
    #undef MAX
    #undef ITEMS_PER_INVOC
    #undef VIRTUAL_WG_SIZE
    #undef DEFINE_ASSIGN

    using ChannelStride = tuple<integral_constant<uint16_t,__padding>,integral_constant<uint16_t,__channelStride_1>,integral_constant<uint16_t,__channelStride_2> >; // we don't use stride 0

    static_assert(VirtualWorkgroupSize<=WorkgroupSize*SubgroupSize);
    static_assert(ItemsPerInvocation_2<=4, "4 level scan would have been needed with this config!");

#ifdef __HLSL_VERSION
    static bool electLast()
    {
        return glsl::gl_SubgroupInvocationID()==SubgroupSize-1;
    }
#endif

    // gets a subgroupID as if each workgroup has (VirtualWorkgroupSize/SubgroupSize) subgroups
    // each subgroup does work (VirtualWorkgroupSize/WorkgroupSize) times, the index denoted by workgroupInVirtualIndex
    static uint16_t virtualSubgroupID(const uint16_t subgroupID, const uint16_t workgroupInVirtualIndex)
    {
        return workgroupInVirtualIndex * (WorkgroupSize >> SubgroupSizeLog2) + subgroupID;
    }

    // get a coalesced index to store for the next level in shared mem, e.g. level 0 -> level 1
    // specify the next level to store values for in template param
    // at level==LevelCount-1, it is guaranteed to have SubgroupSize elements
    template<uint16_t level NBL_FUNC_REQUIRES(level>0 && level<LevelCount)
    static uint16_t sharedStoreIndex(const uint16_t virtualSubgroupID)
    {
        const uint16_t ItemsPerNextInvocation = tuple_element<level,ItemsPerInvocation>::type::value;
        const uint16_t outChannel = virtualSubgroupID & (ItemsPerNextInvocation-uint16_t(1u));
        const uint16_t outInvocation = virtualSubgroupID / ItemsPerNextInvocation;
        const uint16_t localOffset = outChannel * tuple_element<level,ChannelStride>::type::value + outInvocation;

        if (level==2)
        {
            const uint16_t baseOffset = LevelInputCount_1 + (SubgroupSize - uint16_t(1u)) * ItemsPerInvocation_1;
            return baseOffset + localOffset;
        }
        else
        {
            const uint16_t paddingOffset = virtualSubgroupID / (SubgroupSize * ItemsPerInvocation_1);
            return localOffset + paddingOffset;
        }
    }

    template<uint16_t level NBL_FUNC_REQUIRES(level>0 && level<LevelCount)
    static uint16_t sharedStoreIndexFromVirtualIndex(const uint16_t subgroupID, const uint16_t workgroupInVirtualIndex)
    {
        const uint16_t virtualID = virtualSubgroupID(subgroupID, workgroupInVirtualIndex);
        return sharedStoreIndex<level>(virtualID);
    }

    // get the coalesced index in shared mem at the current level
    template<uint16_t level NBL_FUNC_REQUIRES(level>0 && level<LevelCount)
    static uint16_t sharedLoadIndex(const uint16_t invocationIndex, const uint16_t component)
    {
        const uint16_t localOffset = component * tuple_element<level,ChannelStride>::type::value + invocationIndex;
        const uint16_t paddingOffset = invocationIndex / SubgroupSize;

        if (level==2)
        {
            const uint16_t baseOffset = LevelInputCount_1 + (SubgroupSize - uint16_t(1u)) * ItemsPerInvocation_1;
            return baseOffset + localOffset + paddingOffset;
        }
        else
            return localOffset + paddingOffset;
    }
};

#ifndef __HLSL_VERSION
namespace impl
{
struct SVirtualWGSizeLog2
{
    void init(const uint16_t _WorkgroupSizeLog2, const uint16_t _SubgroupSizeLog2)
    {
        #define DEFINE_ASSIGN(TYPE,ID,...) ID = __VA_ARGS__;
        #define MAX(TYPE,ARG1,ARG2) hlsl::max<TYPE>(ARG1, ARG2)
        #define SELECT(TYPE,COND,TRUE_VAL,FALSE_VAL) (COND ? TRUE_VAL : FALSE_VAL)
        #include "impl/virtual_wg_size_def.hlsl"
        #undef SELECT
        #undef MAX
        #undef DEFINE_ASSIGN
    }

    #define DEFINE_ASSIGN(TYPE,ID,...) TYPE ID;
    #include "impl/virtual_wg_size_def.hlsl"
    #undef DEFINE_ASSIGN
};

struct SItemsPerInvoc
{
    void init(const SVirtualWGSizeLog2 virtualWgSizeLog2, const uint16_t BaseItemsPerInvocation)
    {
        #define DEFINE_ASSIGN(TYPE,ID,...) ID = __VA_ARGS__;
        #define VIRTUAL_WG_SIZE virtualWgSizeLog2.
        #define MIN(TYPE,ARG1,ARG2) hlsl::min<TYPE>(ARG1, ARG2)
        #define MAX(TYPE,ARG1,ARG2) hlsl::max<TYPE>(ARG1, ARG2)
        #define SELECT(TYPE,COND,TRUE_VAL,FALSE_VAL) (COND ? TRUE_VAL : FALSE_VAL)
        #include "impl/items_per_invoc_def.hlsl"
        #undef SELECT
        #undef MAX
        #undef MIN
        #undef VIRTUAL_WG_SIZE
        #undef DEFINE_ASSIGN
    }

    #define DEFINE_ASSIGN(TYPE,ID,...) TYPE ID;
    #include "impl/items_per_invoc_def.hlsl"
    #undef DEFINE_ASSIGN
};
}

#include <sstream>
#include <string>
struct SArithmeticConfiguration
{
    void init(const uint16_t _WorkgroupSizeLog2, const uint16_t _SubgroupSizeLog2, const uint16_t _ItemsPerInvocation)
    {
        impl::SVirtualWGSizeLog2 virtualWgSizeLog2;
        virtualWgSizeLog2.init(_WorkgroupSizeLog2, _SubgroupSizeLog2);
        impl::SItemsPerInvoc itemsPerInvoc;
        itemsPerInvoc.init(virtualWgSizeLog2, _ItemsPerInvocation);

        #define DEFINE_ASSIGN(TYPE,ID,...) ID = __VA_ARGS__;
        #define VIRTUAL_WG_SIZE virtualWgSizeLog2.
        #define ITEMS_PER_INVOC itemsPerInvoc.
        #define MAX(TYPE,ARG1,ARG2) hlsl::max<TYPE>(ARG1, ARG2)
        #define SELECT(TYPE,COND,TRUE_VAL,FALSE_VAL) (COND ? TRUE_VAL : FALSE_VAL)
        #include "impl/arithmetic_config_def.hlsl"
        #undef SELECT
        #undef MAX
        #undef ITEMS_PER_INVOC
        #undef VIRTUAL_WG_SIZE
        #undef DEFINE_ASSIGN
    }

    std::string getConfigTemplateStructString()
    {
        std::ostringstream os;
        os << "nbl::hlsl::workgroup2::ArithmeticConfiguration<" << WorkgroupSizeLog2 << "," << SubgroupSizeLog2 << "," << ItemsPerInvocation_0 << ">;";
        return os.str();
    }

    #define DEFINE_ASSIGN(TYPE,ID,...) TYPE ID;
    #include "impl/arithmetic_config_def.hlsl"
    #undef DEFINE_ASSIGN
};
#endif

template<class T>
struct is_configuration : bool_constant<false> {};

template<uint16_t W, uint16_t S, uint16_t I>
struct is_configuration<ArithmeticConfiguration<W,S,I> > : bool_constant<true> {};

template<typename T>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR bool is_configuration_v = is_configuration<T>::value;

}
}
}

#endif
