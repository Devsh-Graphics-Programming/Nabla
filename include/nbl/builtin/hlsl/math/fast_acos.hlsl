// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_FAST_ACOS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FAST_ACOS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{
// https://www.desmos.com/calculator/a59pbwgwof
// a comparison between fast acos methods

// Attempt 1: Polynomial approximation of acos(x) for x in [-1,1]
// Based on an odd cubic fit: acos(x) ~ (a * x^2 + b) * x + pi/2
// Max absolute error: ~9.8e-2 rad
// Very fast: no sqrt, no branches
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct fast_acos_cubic
{
	static T __call(const T val)
	{
		return (T(-0.621565443625098) * val * val - T(0.837561827808302)) * val + numbers::pi<T> / T(2.0);
	}
};

// Attempt 2: Abramowitz & Stegun formula 4.4.45 (adapted for full [-1,1] range)
// acos(x) ~ sqrt(1-x) * (a0 + a1*x + a2*x^2 + a3*x^3) for x in [0,1]
// For x in [-1,0]: acos(x) = pi - acos(-x)
// Max absolute error: ~6.3e-5 rad
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct fast_acos_stegun_poly3
{
	static T __call(const T val)
	{
		const T ax = abs<T>(val);
		const T s = sqrt<T>(T(1.0) - ax);
		const T poly = ((T(-0.019771840941) * ax + T(0.075701735421)) * ax - T(0.212644584569)) * ax + T(1.570771931669); // not pi/2, free constant for better fit
		const T result = s * poly;
		return hlsl::select(val < T(0.0), numbers::pi<T> - result, result);
	}
};

// Attempt 3: Degree-4 polynomial, good accuracy/cost tradeoff
// Fitted as: acos(x) ~ sqrt(1-x) * (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) for x in [0,1]
// Max absolute error: ~8.6e-6 rad
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct fast_acos_stegun_poly4
{
	static T __call(const T val)
	{
		const T ax = abs<T>(val);
		const T s = sqrt<T>(T(1.0) - ax);
		const T poly = (((T(0.0102831457) * ax - T(0.0387865708)) * ax + T(0.0864014439)) * ax - T(0.2144371342)) * ax + numbers::pi<T> / T(2.0);
		const T result = s * poly;
		return hlsl::select(val < T(0.0), numbers::pi<T> - result, result);
	}
};

// Attempt 4: Degree-5 polynomial for best accuracy
// Fitted as: acos(x) ~ sqrt(1-x) * (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5) for x in [0,1]
// Max absolute error: ~1.1e-6 rad
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct fast_acos_stegun_poly5
{
	static T __call(const T val)
	{
		const T ax = abs<T>(val);
		const T s = sqrt<T>(T(1.0) - ax);
		const T poly = ((((T(-0.0051108043) * ax + T(0.0211860706)) * ax - T(0.0464796661)) * ax + T(0.0883884911)) * ax - T(0.2145725267)) * ax + numbers::pi<T> / T(2.0);
		const T result = s * poly;
		return hlsl::select(val < T(0.0), numbers::pi<T> - result, result);
	}
};

template <typename T , int order NBL_STRUCT_CONSTRAINABLE>
struct fast_acos_csc
{
  static T __call(const T val)
  {
    const T u = hlsl::log2(_static_cast<T>(1)+val);
    T poly;
    // See https://www.desmos.com/calculator/sdptomhbju
    if (order==1)
        poly = (_static_cast<T>(1)-u)*_static_cast<T>(0.627);
    else if (order==2)
    {
        const T a = 0.6509;
        const T b = -0.6369;
        const T c = -0.0134;
        poly = hlsl::fma(u, hlsl::fma(u, c, b), a);

    }
    else if (order==3)
    {
        const T a = 0.6494;
        const T b = -0.6310;
        const T c = -0.0123;
        const T d = -0.0004;
        poly = hlsl::fma(u, hlsl::fma(u, hlsl::fma(u, d, c), b), a);
    }
    return hlsl::exp2<T>(poly);
  }
};

template <typename T NBL_STRUCT_CONSTRAINABLE>
struct fast_acos_csc_directed
{
  static T __call(const T val, bool overestimate)
  {
    T u = log2(1.0 + val);

    T a_under = 0.6505590683f;
    T b_under = -0.6369475329f;
    T c_under = -0.0136113553f;

    T a_over = 0.6546594925f;
    T b_over = -0.6269055372f;
    T c_over = -0.0067733255f;

    T a = hlsl::select(overestimate, a_over, a_under);
    T b = hlsl::select(overestimate, b_over, b_under);
    T c = hlsl::select(overestimate, c_over, c_under);
    T poly = hlsl::fma(u, hlsl::fma(u, c, b), a);
    return hlsl::exp2<T>(poly);
  }
};


}
}
}

#endif
