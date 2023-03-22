// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_KERNELS_H_INCLUDED_
#define _NBL_ASSET_KERNELS_H_INCLUDED_


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CFloatingPointSeparableImageFilterKernel.h"

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
