// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_ALPHA_TEST_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_ALPHA_TEST_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{


template <typename PassedPixelsAccessor, typename InCombinedSamplerAccessor>
inline void alpha_test(
	NBL_REF_ARG(PassedPixelsAccessor) passedPixelsAccessor,
	NBL_CONST_REF_ARG(InCombinedSamplerAccessor) inCombinedSamplerAccessor,
	NBL_CONST_REF_ARG(uint16_t3) inDim,
	NBL_CONST_REF_ARG(float32_t) referenceAlpha,
	NBL_CONST_REF_ARG(uint16_t3) globalInvocationID,
	NBL_CONST_REF_ARG(uint16_t3) workGroupID)
{
	if (all(globalInvocationID < inDim))
	{
		const float32_t alpha = inCombinedSamplerAccessor.get(globalInvocationID, workGroupID.z).a;
		if (alpha > referenceAlpha)
		{
			passedPixelsAccessor.AtomicAdd(workGroupID.z, uint32_t(1));
		}
	}
}

}
}
}

#endif

