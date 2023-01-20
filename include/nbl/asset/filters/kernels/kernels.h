// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_KERNELS_H_INCLUDED__
#define __NBL_ASSET_KERNELS_H_INCLUDED__


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CommonImageFilterKernels.h"

namespace nbl::asset
{

// to be inline this function relies on any kernel's `create_sample_functor_t` being defined
template<class CRTP, typename value_type>
template<class PreFilter, class PostFilter>
inline void CImageFilterKernel<CRTP,value_type>::evaluateImpl(
	PreFilter&					preFilter,
	PostFilter&					postFilter,
	value_type*					windowSample,
	core::vectorSIMDf&			relativePos,
	const core::vectorSIMDi32&	globalTexelCoord) const
{
	// static cast is because I'm calling a non-static but non-virtual function
	static_cast<const CRTP*>(this)->create_sample_functor_t(preFilter,postFilter)(windowSample,relativePos,globalTexelCoord, m_multipliedScale);
}

// @see CImageFilterKernel::evaluate
template<class CRTP>
template<class PreFilter, class PostFilter>
inline void CFloatingPointSeparableImageFilterKernelBase<CRTP>::sample_functor_t<PreFilter,PostFilter>::operator()(
		value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const core::vectorSIMDf& multipliedScale)
{
	// this is programmable, but usually in the case of a convolution filter it would be loading the values from a temporary and decoded copy of the input image
	preFilter(windowSample, relativePos, globalTexelCoord, multipliedScale);

	// by default there's no optimization so operation is O(SupportExtent^3) even though the filter is separable
	for (int32_t i=0; i<CRTP::MaxChannels; i++)
	{
		// its possible that the original kernel which defines the `weight` function was stretched or modified, so a correction factor is applied
		windowSample[i] *= (_this->weight(relativePos.x,i)*_this->weight(relativePos.y,i)*_this->weight(relativePos.z,i))* multipliedScale[i];
	}

	// this is programmable, but usually in the case of a convolution filter it would be summing the values
	postFilter(windowSample, relativePos, globalTexelCoord, multipliedScale);
}

} // end namespace nbl::asset

// Kernels
#include "nbl/asset/filters/kernels/CDiracImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CBoxImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CTriangleImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CGaussianImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CKaiserImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CMitchellImageFilterKernel.h"

// Kernel Modifiers
#include "nbl/asset/filters/kernels/CChannelIndependentImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CDerivativeImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CConvolutionImageFilterKernel.h"

#endif