// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"

namespace nbl::asset
{

//! Convert Filter
/*
	Copy while converting format from input image to output image.
	The usage is as follows:
	- create a convert filter reference by \busing YOUR_CONVERT_FILTER = CConvertFormatImageFilter<inputFormat, outputFormat>;\b
	- provide it's state by \bYOUR_CONVERT_FILTER::state_type\b and fill appropriate fields
	- launch one of \bexecute\b calls
	Whenever \binFormat\b or \boutFormat\b passed is \bEF_UNKNOWN\b then the filter uses
	the non-templated runtime conversion variant by looking up the \binFormat\b or \boutFormat\b
	depending on which is \bEF_UNKNOWN\b from the \binImage\b or \boutImage\b stored in filter's state
	creation parameters.

	@see IImageFilter
	@see CSwizzleAndConvertImageFilter
*/

// copy while converting format from input image to output image
template<E_FORMAT inFormat=EF_UNKNOWN, E_FORMAT outFormat=EF_UNKNOWN, class Dither=IdentityDither, typename Normalization=void, bool Clamp=false>
class CConvertFormatImageFilter : public CSwizzleAndConvertImageFilter<inFormat,outFormat,VoidSwizzle,Dither,Normalization,Clamp>
{
	public:
		virtual ~CConvertFormatImageFilter() {}
		
		using state_type = typename CSwizzleAndConvertImageFilter<inFormat,outFormat,VoidSwizzle,Dither,Normalization,Clamp>::state_type;
};

} // end namespace nbl::asset

#endif