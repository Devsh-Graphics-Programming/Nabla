// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_SCRATCH_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_SCRATCH_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic_portability.hlsl"

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
		 * Struct creator with default values for subgroup portability arithmetic operations.
		 * For each subgroup inside the workgroup, the memory range is SubgroupSize + HalfSubgroupSize
		 * because each subgroup has HalfSubgroupSize padding on its left. This means that the ranges 
		 * for a 4-subgroup workgroup with size 64 are:
		 * [0,95] [96,191] [192,287] [288,383].
		 * The padding is usually initialized with an operation's identity value while the main range 
		 * with the operated value.
		 */
		static ScratchOffsetsAndMasks WithSubgroupOpDefaults()
		{
			ScratchOffsetsAndMasks s;
			s.subgroupMask = Size() - 1u;
			s.halfSubgroupSize = Size() >> 1u; // this is also the size of the padding memory
			s.subgroupInvocation = InvocationID();
			s.subgroupId = SubgroupID();
			s.subgroupElectedLocalInvocation = ElectedLocalInvocationID();
			s.lastSubgroupInvocation = s.subgroupMask;
			if(s.subgroupId == ((_NBL_HLSL_WORKGROUP_SIZE_ - 1u) >> SizeLog2())) {
				s.lastSubgroupInvocation &= _NBL_HLSL_WORKGROUP_SIZE_ - 1u; // if the workgroup size is not a power of 2, then the lastSubgroupInvocation for the last subgroup of the workgroup will not be equal to the subgroupMask but something smaller
			}
			s.subgroupMemoryBegin = s.subgroupElectedLocalInvocation + s.halfSubgroupSize * (s.subgroupElectedLocalInvocation >> SizeLog2());
			s.lastLoadOffset = s.subgroupMemoryBegin + s.subgroupInvocation;
			s.subgroupPaddingMemoryEnd = s.subgroupMemoryBegin + s.halfSubgroupSize;
			s.scanStoreOffset = s.subgroupPaddingMemoryEnd + s.subgroupInvocation;
			return s;
		}
	
		uint subgroupMask;
		uint halfSubgroupSize;
		uint subgroupInvocation;
		uint lastSubgroupInvocation;
		uint subgroupId;
		uint subgroupElectedLocalInvocation;
		uint subgroupMemoryBegin;
		uint lastLoadOffset;
		uint scanStoreOffset;
		
		uint subgroupPaddingMemoryEnd;
	};
	
	template<class ScratchAccessor, typename T>
	void scratchInitialize(T value, T identity, uint activeInvocationIndexUpperBound)
	{
		ScratchAccessor scratch;
		ScratchOffsetsAndMasks offsetsAndMasks = ScratchOffsetsAndMasks::WithSubgroupOpDefaults();
		scratch.main.set(offsetsAndMasks.scanStoreOffset, value);
		const uint halfSubgroupMask = offsetsAndMasks.subgroupMask >> 1u;
		
		//OLD CODE WAS USING THIS -> scratch.main.set(scratchInitializeClearIndex(gl_LocalInvocationIndex, halfSubgroupMask), identity);
		if (offsetsAndMasks.subgroupInvocation < offsetsAndMasks.halfSubgroupSize) {
			scratch.main.set(offsetsAndMasks.lastLoadOffset, identity);
		}
		
		bool isLastSubgroupInWG = ((_NBL_HLSL_WORKGROUP_SIZE_-1u) >> SizeLog2()) == offsetsAndMasks.subgroupId;
		uint lastSubgroupSize = offsetsAndMasks.lastSubgroupInvocation + 1;
		if(isLastSubgroupInWG && lastSubgroupSize < offsetsAndMasks.halfSubgroupSize) {
			// In this case, the workgroup size is such that the last subgroup is smaller than halfSubgroupSize.
			// This means that some of the padding memory, which we initialize with the identity value, will 
			// remain uninitialized. We must do more initializations with the subgroup invocations we have.
			uint llo = offsetsAndMasks.lastLoadOffset;
			uint padEnd = offsetsAndMasks.subgroupPaddingMemoryEnd;
			for(uint ix = 1; llo + (ix * lastSubgroupSize) < padEnd; ix++) {
				scratch.main.set(offsetsAndMasks.lastLoadOffset + (ix * lastSubgroupSize), identity);
			}
		}
		
		if(_NBL_HLSL_WORKGROUP_SIZE_ < offsetsAndMasks.halfSubgroupSize)
		{
			const uint maxItemsToClear = (((activeInvocationIndexUpperBound - 1u) & (~offsetsAndMasks.subgroupMask)) >> 1u) + offsetsAndMasks.halfSubgroupSize;
			for (uint ix = gl_LocalInvocationIndex + _NBL_HLSL_WORKGROUP_SIZE_; ix < maxItemsToClear; ix += _NBL_HLSL_WORKGROUP_SIZE_) {
				scratch.main.set(scratchInitializeClearIndex(ix, halfSubgroupMask), identity);
			}
		}
		workgroup::Barrier();
	}
}
}
}

#endif