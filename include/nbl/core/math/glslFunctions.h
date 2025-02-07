// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_GLSL_FUNCTIONS_H_INCLUDED__
#define __NBL_CORE_GLSL_FUNCTIONS_H_INCLUDED__

#include <type_traits>
#include <utility>

#include "nbl/type_traits.h"
#include "nbl/core/math/floatutil.h"

namespace nbl
{
namespace core
{

template<typename T>
NBL_FORCE_INLINE T sinc(const T& x)
{
	// TODO: do a direct series/computation in the future
	return mix<T>(sin<T>(x) / x,
		T(1.0) + x * x * (x * x * T(1.0 / 120.0) - T(1.0 / 6.0)),
		abs<T>(x) < T(0.0001)
	);
}
template<typename T>
NBL_FORCE_INLINE T d_sinc(const T& x)
{
	// TODO: do a direct series/computation in the future
	return mix<T>((cos<T>(x) - sin<T>(x) / x) / x,
		x * (x * x * T(4.0 / 120.0) - T(2.0 / 6.0)),
		abs<T>(x) < T(0.0001)
	);
}

template<typename T>
NBL_FORCE_INLINE T cyl_bessel_i(const T& v, const T& x);
template<typename T>
NBL_FORCE_INLINE T d_cyl_bessel_i(const T& v, const T& x);

template<typename T>
NBL_FORCE_INLINE T KaiserWindow(const T& x, const T& alpha, const T& width)
{
	auto p = x/width;
	return cyl_bessel_i<T>(T(0.0),sqrt<T>(T(1.0)-p*p)*alpha)/cyl_bessel_i<T>(T(0.0),alpha);
}
template<typename T>
NBL_FORCE_INLINE T d_KaiserWindow(const T& x, const T& alpha, const T& width)
{
	auto p = x/width;
	T s = sqrt<T>(T(1.0)-p*p);
	T u = s*alpha;
	T du = -p*alpha/(width*s);
	return du*d_cyl_bessel_i<T>(T(0.0),u)/cyl_bessel_i<T>(T(0.0),alpha);
}


} // end namespace core
} // end namespace nbl

#endif

