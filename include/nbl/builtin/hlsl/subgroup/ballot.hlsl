// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{

uint32_t LastSubgroupInvocation()
{
    // why this code was wrong before:
    // - only compute can use SubgroupID
    // - but there's no mapping of InvocationID to SubgroupID and Index
    return glsl::subgroupBallotFindMSB(glsl::subgroupBallot(true));
}

bool ElectLast()
{
    return glsl::gl_SubgroupInvocationID()==LastSubgroupInvocation();
}

template<typename T>
T BroadcastLast(T value)
{
    return glsl::subgroupBroadcast<T>(value,LastSubgroupInvocation());
}

uint32_t ElectedSubgroupInvocationID() {
    return glsl::subgroupBroadcastFirst<uint32_t>(glsl::gl_SubgroupInvocationID());
}

template<uint32_t SubgroupSizeLog2>
struct Configuration
{
    using mask_t = conditional_t<SubgroupSizeLog2 < 7, conditional_t<SubgroupSizeLog2 < 6, uint32_t1, uint32_t2>, uint32_t4>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t Size = 0x1u << SubgroupSizeLog2;
};

template<class T>
struct is_configuration : bool_constant<false> {};

template<uint32_t N>
struct is_configuration<Configuration<N> > : bool_constant<true> {};

template<typename T>
NBL_CONSTEXPR bool is_configuration_v = is_configuration<T>::value;

}
}
}

#endif