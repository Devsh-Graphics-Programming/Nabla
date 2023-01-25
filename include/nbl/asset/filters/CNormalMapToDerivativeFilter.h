// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED_
#define _NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED_

#include "nbl/core/pch_core.h"

#include <type_traits>
#include <functional>
 
#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "CConvertFormatImageFilter.h"

namespace nbl::asset
{

/*
	NormalMap to DerivativeMap Swizzle
*/
struct NBL_API NormalMapToDerivativeMapSwizzle
{
	// since most normalmaps are supplied in RG8_UNORM
	double zeroEpsilon = 1.0 / 255.0;

	template<typename InT, typename OutT>
	void operator()(const InT* in, OutT* out) const
	{
		const auto* _in = reinterpret_cast<const std::conditional_t<std::is_void_v<InT>,uint64_t,InT>*>(in);
		auto* _out = reinterpret_cast<std::conditional_t<std::is_void_v<OutT>,uint64_t,OutT>*>(out);
		const auto xDecode = _in[0]*2.f-1.f;
		const auto yDecode = _in[1]*2.f-1.f;
		// TODO: a template parameter to decide if Z is in [0,1] or [-1,1]
		const auto zDecode = _in[2]*2.f-1.f;
		// because normalmaps are supplied in UNORM formats, there's no true zero
		_out[0] = core::abs(xDecode) > zeroEpsilon ? (-xDecode / zDecode) : 0.f;
		// scanlines go from top down, so Y component is in reverse
		_out[1] = core::abs(yDecode) > zeroEpsilon ? (yDecode / zDecode) : 0.f;
	}
};

template<bool isotropic>
using CNormalMapToDerivativeFilter = CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,NormalMapToDerivativeMapSwizzle,IdentityDither,CDerivativeMapNormalizationState<isotropic>,true>;

} // end namespace nbl::asset

#endif // __NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED__