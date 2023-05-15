// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_SCRATCH_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_SCRATCH_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic_portability.hlsl"

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex;
#endif

/**
 * The clearIndex for each invocation is the first 8 indices for each set of subgroup::Size().
 * Example for subgroup::Size() == 32:
 *	invocationId -> clearIndex
 *	1 -> 1
 *	7 -> 7
 *	8 -> 32
 *	15 -> 39
 *	16 -> 64
 *	24 -> 96
 */
inline uint scratchInitializeClearIndex(uint invocationId, uint halfSubgroupMask)
{
	return ((((invocationId)&(~halfSubgroupMask))<<2u)|((invocationId)&halfSubgroupMask));
}

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
	struct ScratchOffsetsAndMasks
	{
		/**
		 * Struct creator with default values.
		 */
		static ScratchOffsetsAndMasks WithDefaults()
		{
			ScratchOffsetsAndMasks s;
			s.subgroupMask = Size() - 1u;
			s.halfSubgroupSize = Size() >> 1u;
			s.subgroupInvocation = InvocationID();
			s.subgroupElectedInvocation = Elect();
			s.subgroupMemoryBegin = s.subgroupElectedInvocation << 1u;
			
			// REVIEW: In the glsl implementation these are initialized like so
			s.lastLoadOffset = s.subgroupMemoryBegin | s.subgroupInvocation;
			s.scanStoreOffset = s.lastLoadOffset + s.halfSubgroupSize;
			// REVIEW: In the comment https://github.com/Devsh-Graphics-Programming/Nabla/pull/434#issuecomment-1312824804 these are initialized as below
			// lastLoadOffset = subgroupMemoryBegin+subgroupInvocation;
			// paddingMemoryEnd = subgroupMemoryBegin+halfSubgroupSize;
			// scanStoreOffset = paddingMemoryEnd+subgroupInvocation;
			// REVIEW: Which is correct?
			
			s.paddingMemoryEnd = s.subgroupMemoryBegin + s.halfSubgroupSize;
			
			return s;
		}
	
		uint subgroupMask;
		uint halfSubgroupSize;
		uint subgroupInvocation;
		uint subgroupElectedInvocation;
		uint subgroupMemoryBegin;
		uint lastLoadOffset;
		uint scanStoreOffset;
		
		uint paddingMemoryEnd;
	};
	
	template<class ScratchAccessor, typename T>
	void scratchInitialize(T value, T identity, uint activeInvocationIndexUpperBound)
	{
		ScratchAccessor accessor;
		ScratchOffsetsAndMasks offsetsAndMasks = ScratchOffsetsAndMasks::WithDefaults();
		accessor.set(offsetsAndMasks.scanStoreOffset, value);
		const uint halfSubgroupMask = offsetsAndMasks.subgroupMask >> 1u;
		accessor.set(scratchInitializeClearIndex(gl_LocalInvocationIndex, halfSubgroupMask), identity);
		
		if(_NBL_HLSL_WORKGROUP_SIZE_ < offsetsAndMasks.halfSubgroupSize)
		{
			const uint maxItemsToClear = (((activeInvocationIndexUpperBound - 1u) & (~offsetsAndMasks.subgroupMask)) >> 1u) + offsetsAndMasks.halfSubgroupSize;
			for (uint ix = gl_LocalInvocationIndex + _NBL_HLSL_WORKGROUP_SIZE_; ix < maxItemsToClear; ix += _NBL_HLSL_WORKGROUP_SIZE_)
				accessor.set(scratchInitializeClearIndex(ix, halfSubgroupMask), identity);
		}
		workgroup::Barrier();
	}
}
}
}

#endif