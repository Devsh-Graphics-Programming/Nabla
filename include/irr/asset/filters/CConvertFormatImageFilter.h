// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CSwizzleAndConvertImageFilter.h"

namespace irr
{
namespace asset
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
template<E_FORMAT inFormat=EF_UNKNOWN, E_FORMAT outFormat=EF_UNKNOWN, bool Normalize = false, bool Clamp = false, class Dither = IdentityDither>
class CConvertFormatImageFilter : public CSwizzleAndConvertImageFilter<inFormat,outFormat,VoidSwizzle,Normalize,Clamp,Dither>
{
	public:
		virtual ~CConvertFormatImageFilter() {}
		
		using state_type = typename CSwizzleAndConvertImageFilter<inFormat,outFormat,VoidSwizzle,Normalize,Clamp,Dither>::state_type;
};

} // end namespace asset
} // end namespace irr

#endif