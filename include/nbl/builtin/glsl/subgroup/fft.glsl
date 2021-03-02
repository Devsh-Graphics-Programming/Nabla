// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_FFT_INCLUDED_


#include <nbl/builtin/glsl/subgroup/shared_shuffle_portability.glsl>

#include <nbl/builtin/glsl/math/complex.glsl>
#include <nbl/builtin/glsl/subgroup/shuffle_portability.glsl>


//TODO: optimization for DFT of real signal

// TODO: with stockham or something that does not require stupid shuffles to extract and pack
void nbl_glsl_subgroupFFT_loop(in bool is_inverse, in uint stride, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    const uint sub_ix = nbl_glsl_SubgroupInvocationID&(stride-1u);
    nbl_glsl_subgroupBarrier();
    nbl_glsl_complex low = nbl_glsl_subgroupShuffleXor(lo,stride);
    nbl_glsl_complex high = nbl_glsl_subgroupShuffleXor(hi);
    
    nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(is_inverse,sub_ix,float(stride<<1u));
    if (is_inverse)
        nbl_glsl_FFT_DIT_radix2(twiddle,low,high);
    else
        nbl_glsl_FFT_DIF_radix2(twiddle,low,high);

    nbl_glsl_subgroupBarrier();
    lo = low;
    hi = high;
}
// Decimates in Frequency for forward transform, in Time for reverse, hence no bitreverse permutation needed
void nbl_glsl_subgroupFFT(in bool is_inverse, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    const float doubleSubgroupSize = float(nbl_glsl_SubgroupSize<<1u);
    // special first iteration
	if (!is_inverse)
        nbl_glsl_FFT_DIF_radix2(nbl_glsl_FFT_twiddle(false,nbl_glsl_SubgroupInvocationID,doubleSubgroupSize),lo,hi);

	if (is_inverse)
    for (uint step=1u; step<_NBL_GLSL_WORKGROUP_SIZE_; step<<=1u)
        nbl_glsl_subgroupFFT_loop(true,step,lo,hi);
	else
    for (uint step=nbl_glsl_SubgroupSize>>1u; step>0u; step>>=1u)
        nbl_glsl_subgroupFFT_loop(false,step,lo,hi);

    nbl_glsl_subgroupBarrier();
	if (is_inverse)
	{
        nbl_glsl_FFT_DIT_radix2(nbl_glsl_FFT_twiddle(true,nbl_glsl_SubgroupInvocationID,doubleSubgroupSize),lo,hi);

        lo /= doubleSubgroupSize;
        hi /= doubleSubgroupSize;
	}
}

#endif