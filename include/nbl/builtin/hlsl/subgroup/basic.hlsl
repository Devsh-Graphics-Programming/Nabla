// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_

// TODO (PentaKon): All of these have to be ported to spirv intrinsics

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
	
uint LastSubgroupInvocation() {
	uint lastSubgroupInvocation = Size() - 1u;
	if(SubgroupID() == ((_NBL_HLSL_WORKGROUP_SIZE_ - 1u) >> SizeLog2())) {
		lastSubgroupInvocation &= _NBL_HLSL_WORKGROUP_SIZE_ - 1u; // if the workgroup size is not a power of 2, then the lastSubgroupInvocation for the last subgroup of the workgroup will not be equal to the subgroupMask but something smaller
	}
	return lastSubgroupInvocation;
}

}
}
}

#endif