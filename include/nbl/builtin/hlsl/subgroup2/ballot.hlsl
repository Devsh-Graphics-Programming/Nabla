// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup2
{

template<int32_t AssumeAllActive=false>
uint32_t LastSubgroupInvocation()
{
    if (AssumeAllActive)
        return glsl::gl_SubgroupSize()-1;
    else
        return glsl::subgroupBallotFindMSB(glsl::subgroupBallot(true));
}

bool ElectLast()
{
    return glsl::gl_SubgroupInvocationID()==LastSubgroupInvocation();
}

template<uint32_t SubgroupSizeLog2>
struct Configuration
{
    using mask_t = conditional_t<SubgroupSizeLog2 < 7, conditional_t<SubgroupSizeLog2 < 6, uint32_t1, uint32_t2>, uint32_t4>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t SizeLog2 = uint16_t(SubgroupSizeLog2);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Size = uint16_t(0x1u) << SubgroupSizeLog2;
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
