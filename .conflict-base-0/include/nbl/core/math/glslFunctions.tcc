// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_GLSL_FUNCTIONS_TCC_INCLUDED__
#define __NBL_CORE_GLSL_FUNCTIONS_TCC_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/core/math/floatutil.tcc"
#include "matrix4SIMD.h"

#include <cmath>
#include <numeric>

namespace nbl
{
namespace core
{

template<>
NBL_FORCE_INLINE vectorSIMDf abs<vectorSIMDf>(const vectorSIMDf& a)
{
    return a&vectorSIMD_32<uint32_t>(0x7FFFFFFFu,0x7FFFFFFFu,0x7FFFFFFFu,0x7FFFFFFFu);
}
template<typename T>
NBL_FORCE_INLINE T abs(const T& a)
{
	auto isneg = a < T(0);
	using bool_type = std::remove_reference_t<decltype(isneg)>;
	return core::mix<T,bool_type>(a,-a,isneg);
}


template<>
NBL_FORCE_INLINE float sqrt<float>(const float& x)
{
	return std::sqrt(x); // can't use sqrtf on sqrt
}
template<>
NBL_FORCE_INLINE double sqrt<double>(const double& x)
{
	return std::sqrt(x);
}
template<>
NBL_FORCE_INLINE vectorSIMDf sqrt(const vectorSIMDf& x)
{
#ifdef __NBL_COMPILE_WITH_X86_SIMD_
	return _mm_sqrt_ps(x.getAsRegister());
#else
#error "no implementation"
#endif
}

template<>
NBL_FORCE_INLINE vectorSIMDf inversesqrt<vectorSIMDf>(const vectorSIMDf& x)
{
#ifdef __NBL_COMPILE_WITH_X86_SIMD_
	return _mm_rsqrt_ps(x.getAsRegister());
#else
#error "no implementation"
#endif
}
template<typename T>
NBL_FORCE_INLINE T inversesqrt(const T& x)
{
	return core::reciprocal<T>(core::sqrt<T>(x));
}

template<>
NBL_FORCE_INLINE vectorSIMDf reciprocal_approxim<vectorSIMDf>(const vectorSIMDf& x)
{
#ifdef __NBL_COMPILE_WITH_X86_SIMD_
    return _mm_rcp_ps(x.getAsRegister());
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE float reciprocal_approxim<float>(const float& x)
{
#if defined (__NBL_FAST_MATH) && defined ( __NBL_COMPILE_WITH_X86_SIMD_ )
    float result;
    _mm_store_ss(&result,_mm_rcp_ps(_mm_load_ss(&x)));
    return result;
#else // no fast math
	return reciprocal<float>(x);
#endif
}
template<typename T>
NBL_FORCE_INLINE T reciprocal_approxim(const T& x)
{
    return reciprocal<T>(x);
}

template<>
NBL_FORCE_INLINE float exp2<float>(const float& x)
{
	return std::exp2f(x);
}
template<>
NBL_FORCE_INLINE double exp2<double>(const double& x)
{
	return std::exp2(x);
}



template<>
NBL_FORCE_INLINE float floor<float>(const float& x)
{
	return std::floor(x); // can't use the f-suffix on GCC, doesn't compile
}
template<>
NBL_FORCE_INLINE double floor<double>(const double& x)
{
	return std::floor(x);
}
template<>
NBL_FORCE_INLINE vectorSIMDf floor<vectorSIMDf>(const vectorSIMDf& x)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_floor_ps(x.getAsRegister());
#else
#error "no implementation"
#endif
}

/*
//! round towards zero
template<typename T>
NBL_FORCE_INLINE T trunc(const T& x);


//! No idea how this shit works, unimplemented
template<typename T>
NBL_FORCE_INLINE T roundEven(const T& x);
*/

template<>
NBL_FORCE_INLINE float ceil<float>(const float& x)
{
	return std::ceil(x); // can't use the f-suffix on GCC, doesn't compile
}
template<>
NBL_FORCE_INLINE double ceil<double>(const double& x)
{
	return std::ceil(x);
}
template<>
NBL_FORCE_INLINE vectorSIMDf ceil<vectorSIMDf>(const vectorSIMDf& x)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_ceil_ps(x.getAsRegister());
#else
#error "no implementation"
#endif
}

template<>
NBL_FORCE_INLINE vectorSIMDf min<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_min_ps(a.getAsRegister(),b.getAsRegister());
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE vectorSIMDu32 min<vectorSIMDu32>(const vectorSIMDu32& a, const vectorSIMDu32& b)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_min_epu32(a.getAsRegister(),b.getAsRegister());
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE vectorSIMDi32 min<vectorSIMDi32>(const vectorSIMDi32& a, const vectorSIMDi32& b)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_min_epi32(a.getAsRegister(),b.getAsRegister());
#else
#error "no implementation"
#endif
}
template<class T>
NBL_FORCE_INLINE T min(const T& a, const T& b)
{
	T vb = T(b);
	auto asmaller = a<vb;
	using bool_type = std::remove_reference_t<decltype(asmaller)>;
	return core::mix<T,bool_type>(vb,a,asmaller);
}

template<>
NBL_FORCE_INLINE vectorSIMDf max<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_max_ps(a.getAsRegister(),b.getAsRegister());
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE vectorSIMDu32 max<vectorSIMDu32>(const vectorSIMDu32& a, const vectorSIMDu32& b)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_max_epu32(a.getAsRegister(),b.getAsRegister());
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE vectorSIMDi32 max<vectorSIMDi32>(const vectorSIMDi32& a, const vectorSIMDi32& b)
{
#ifdef __NBL_COMPILE_WITH_SSE3
	return _mm_max_epi32(a.getAsRegister(),b.getAsRegister());
#else
#error "no implementation"
#endif
}
template<class T>
NBL_FORCE_INLINE T max(const T& a, const T& b)
{
	T vb = T(b);
	auto asmaller = a < vb;
	using bool_type = std::remove_reference_t<decltype(asmaller)>;
	return core::mix<T,bool_type>(a,vb,asmaller);
}


template<>
NBL_FORCE_INLINE vectorSIMDf dot<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b)
{
    __m128 xmm0 = a.getAsRegister();
    __m128 xmm1 = b.getAsRegister();
#ifdef __NBL_COMPILE_WITH_SSE3
    xmm0 = _mm_mul_ps(xmm0,xmm1);
    xmm0 = _mm_hadd_ps(xmm0,xmm0);
    return _mm_hadd_ps(xmm0,xmm0);
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE vectorSIMDi32 dot<vectorSIMDi32>(const vectorSIMDi32& a, const vectorSIMDi32& b)
{
    __m128i xmm0 = (a*b).getAsRegister();
#ifdef __NBL_COMPILE_WITH_SSE3
    xmm0 = _mm_hadd_epi32(xmm0,xmm0);
    return _mm_hadd_epi32(xmm0,xmm0);
#else
#error "no implementation"
#endif
}
template<>
NBL_FORCE_INLINE vectorSIMDu32 dot<vectorSIMDu32>(const vectorSIMDu32& a, const vectorSIMDu32& b)
{
    __m128i xmm0 = (a*b).getAsRegister();
#ifdef __NBL_COMPILE_WITH_SSE3
    xmm0 = _mm_hadd_epi32(xmm0,xmm0);
    return _mm_hadd_epi32(xmm0,xmm0);
#else
#error "no implementation"
#endif
}

template<>
NBL_FORCE_INLINE vectorSIMDf cross<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b)
{
#ifdef __NBL_COMPILE_WITH_X86_SIMD_
    __m128 xmm0 = a.getAsRegister();
    __m128 xmm1 = b.getAsRegister();
    __m128 backslash = _mm_mul_ps(FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,0,2,1)),FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,1,0,2)));
    __m128 forwardslash = _mm_mul_ps(FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,1,0,2)),FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,0,2,1)));
    return _mm_sub_ps(backslash,forwardslash); //returns 0 in the last component :D
#else
#error "no implementation"
#endif
}

template<>
NBL_FORCE_INLINE matrix4SIMD transpose(const matrix4SIMD& m)
{
	core::matrix4SIMD retval;
	__m128 a0 = m.rows[0].getAsRegister(), a1 = m.rows[1].getAsRegister(), a2 = m.rows[2].getAsRegister(), a3 = m.rows[3].getAsRegister();
	_MM_TRANSPOSE4_PS(a0, a1, a2, a3);
	retval.rows[0] = a0;
	retval.rows[1] = a1;
	retval.rows[2] = a2;
	retval.rows[3] = a3;
	return retval;
}



template<>
NBL_FORCE_INLINE bool equals<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& tolerance)
{
	return ((a + tolerance >= b) && (a - tolerance <= b)).all();
}
template<>
NBL_FORCE_INLINE bool equals(const core::vector3df& a, const core::vector3df& b, const core::vector3df& tolerance)
{
	auto ha = a+tolerance;
	auto la = a-tolerance;
	return ha.X>=b.X&&ha.Y>=b.Y&&ha.Z>=b.Z && la.X<=b.X&&la.Y<=b.Y&&la.Z<=b.Z;
}
template<>
NBL_FORCE_INLINE bool equals<matrix4SIMD>(const matrix4SIMD& a, const matrix4SIMD& b, const matrix4SIMD& tolerance)
{
	for (size_t i = 0u; i<matrix4SIMD::VectorCount; ++i)
		if (!equals<vectorSIMDf>(a.rows[i], b.rows[i], tolerance.rows[i]))
			return false;
	return true;
}
template<>
NBL_FORCE_INLINE bool equals<matrix3x4SIMD>(const matrix3x4SIMD& a, const matrix3x4SIMD& b, const matrix3x4SIMD& tolerance)
{
	for (size_t i = 0u; i<matrix3x4SIMD::VectorCount; ++i)
		if (!equals<vectorSIMDf>(a.rows[i], b.rows[i], tolerance[i]))
			return false;
	return true;
}
template<typename T>
NBL_FORCE_INLINE bool equals(const T& a, const T& b, const T& tolerance)
{
	return (a + tolerance >= b) && (a - tolerance <= b);
}


template<>
NBL_FORCE_INLINE vectorSIMDf sin<vectorSIMDf>(const vectorSIMDf& a)
{
	// TODO: vastly improve this
	return vectorSIMDf(sin<float>(a.x),sin<float>(a.y),sin<float>(a.z),sin<float>(a.w));
}
template<typename T>
NBL_FORCE_INLINE T sin(const T& a)
{
	return std::sin(a);
}



// extras


template<>
NBL_FORCE_INLINE vectorSIMDu32 gcd<vectorSIMDu32>(const vectorSIMDu32& a, const vectorSIMDu32& b)
{
	return vectorSIMDu32(gcd<uint32_t>(a.x,b.x),gcd<uint32_t>(a.y,b.y),gcd<uint32_t>(a.z,b.z),gcd<uint32_t>(a.w,b.w));
}
template<typename T>
NBL_FORCE_INLINE T gcd(const T& a, const T& b)
{
	return std::gcd(a,b);
}

// https://libcxx.llvm.org/docs/Cxx1zStatus.html "Mathematical Special Functions for C++17"
#ifndef _NBL_PLATFORM_ANDROID_
template<>
NBL_FORCE_INLINE vectorSIMDf cyl_bessel_i<vectorSIMDf>(const vectorSIMDf& v, const vectorSIMDf& x)
{
	return vectorSIMDf(cyl_bessel_i<float>(v[0],x[0]),cyl_bessel_i<float>(v[1],x[1]),cyl_bessel_i<float>(v[2],x[2]),cyl_bessel_i<float>(v[3],x[3]));
}
template<typename T>
NBL_FORCE_INLINE T cyl_bessel_i(const T& v, const T& x)
{
	return std::cyl_bessel_i(double(v),double(x));
}

template<>
NBL_FORCE_INLINE vectorSIMDf d_cyl_bessel_i<vectorSIMDf>(const vectorSIMDf& v, const vectorSIMDf& x)
{
	return vectorSIMDf(d_cyl_bessel_i<float>(v[0],x[0]),d_cyl_bessel_i<float>(v[1],x[1]),d_cyl_bessel_i<float>(v[2],x[2]),d_cyl_bessel_i<float>(v[3],x[3]));
}
template<typename T>
NBL_FORCE_INLINE T d_cyl_bessel_i(const T& v, const T& x)
{
	return 0.5*(std::cyl_bessel_i(double(v)-1.0,double(x))+std::cyl_bessel_i(double(v)+1.0,double(x)));
}
#endif


} // end namespace core
} // end namespace nbl

#endif

