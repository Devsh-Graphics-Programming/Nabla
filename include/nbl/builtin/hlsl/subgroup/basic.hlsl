// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
	
static const uint MinSubgroupSizeLog2 = 4;
static const uint MinSubgroupSize = 0x1u << MinSubgroupSizeLog2;
	
uint LastSubgroupInvocation() {
	uint lastSubgroupInvocation = glsl::gl_SubgroupSize() - 1u;
	if(glsl::gl_SubgroupID() == ((_NBL_HLSL_WORKGROUP_SIZE_ - 1u) >> glsl::gl_SubgroupSizeLog2())) {
		lastSubgroupInvocation &= _NBL_HLSL_WORKGROUP_SIZE_ - 1u; // if the workgroup size is not a power of 2, then the lastSubgroupInvocation for the last subgroup of the workgroup will not be equal to the subgroupMask but something smaller
	}
	return lastSubgroupInvocation;
}

}
}
}

#endif