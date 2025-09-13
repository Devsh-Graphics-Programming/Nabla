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

template<int components>
class vectorSIMDBool;
template <class T>
class vectorSIMD_32;
class vectorSIMDf;
class matrix4SIMD;
class matrix3x4SIMD;


template<typename T>
NBL_FORCE_INLINE T radians(const T& degrees)
{
	//static_assert(
	//	std::is_same<T, float>::value ||
	//	std::is_same<T, double>::value ||
	//	std::is_same<T, vectorSIMDf>::value,
	//	"This code expects the type to be double, float or vectorSIMDf, only (float, double, vectorSIMDf).");

	return degrees*PI<T>()/T(180);
}
template<typename T>
NBL_FORCE_INLINE T degrees(const T& radians)
{
	static_assert(
		std::is_same<T, float>::value ||
		std::is_same<T, double>::value ||
		std::is_same<T, vectorSIMDf>::value,
		"This code expects the type to be double, float or vectorSIMDf, only (float, double, vectorSIMDf).");

	return radians*T(180)/PI<T>();
}
// TODO : sin,cos,tan,asin,acos,atan(y,x),atan(y_over_x),sinh,cosh,tanh,asinh,acosh,atanh,sincos specializations for vectors
template<typename T>
T sin(const T& radians);
template<typename T>
NBL_FORCE_INLINE T cos(const T& radians)
{
	return std::cos(radians);
}

// TODO : pow,exp,log,exp2,log2
template<typename T>
NBL_FORCE_INLINE T sqrt(const T& x);
template<>
NBL_FORCE_INLINE float sqrt<float>(const float& x);
template<>
NBL_FORCE_INLINE double sqrt<double>(const double& x);
template<>
NBL_FORCE_INLINE vectorSIMDf sqrt<vectorSIMDf>(const vectorSIMDf& x);

template<typename T>
NBL_FORCE_INLINE T inversesqrt(const T& x);

template<typename T>
NBL_FORCE_INLINE T reciprocal(const T& x)
{
	return T(1.0)/x;
}

template<typename T>
NBL_FORCE_INLINE T reciprocal_approxim(const T& x);
template<>
NBL_FORCE_INLINE float reciprocal_approxim<float>(const float& x);
template<>
NBL_FORCE_INLINE vectorSIMDf reciprocal_approxim<vectorSIMDf>(const vectorSIMDf& x);

template<typename T>
NBL_FORCE_INLINE T exp2(const T& x);
template<>
NBL_FORCE_INLINE float exp2<float>(const float& x);
template<>
NBL_FORCE_INLINE double exp2<double>(const double& x);

//! TODO : find some intrinsics
template<typename T>
NBL_FORCE_INLINE T fma(const T& a, const T& b, const T& c)
{
	return a * b + c;
}

//! returns linear interpolation of a and b with ratio t
//! only defined for floating point scalar and vector types
//! \return: a if t==0, b if t==1, and the linear interpolation else
template<typename T, typename U>
NBL_FORCE_INLINE T mix(const T & a, const T & b, const U & t)
{
	T retval;
	if constexpr(nbl::is_any_of<U,vectorSIMDBool<2>,vectorSIMDBool<4>,vectorSIMDBool<8>,vectorSIMDBool<16> >::value)
	{
		if constexpr(std::is_same<T,vectorSIMDf>::value)
		{
			retval = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128((a&(~t)).getAsRegister()),_mm_castps_si128((b&t).getAsRegister())));
		}
		else
		{
			retval = (a&(~t))|(b&t);
		}
		
	}
	else
	{
		if constexpr(std::is_same<U,bool>::value)
		{
			retval = t ? b:a;
		}
		else
		{
			if constexpr(nbl::is_any_of<T,matrix4SIMD,matrix3x4SIMD>::value)
			{
				for (uint32_t i=0u; i<T::VectorCount; i++)
				{
					if constexpr(nbl::is_any_of<U, matrix4SIMD, matrix3x4SIMD>::value)
					{
						retval[i] = core::mix<vectorSIMDf, vectorSIMDf>(a.rows[i], b.rows[i], t.rows[i]);
					}
					else
					{
						retval[i] = core::mix<vectorSIMDf, U>(a.rows[i], b.rows[i], t);
					}
					
				}
			}
			else
			{
				retval = core::fma<T>(b-a,t,a);
			}
			
		}
		
	}
	
	return retval;
}

//! returns abs of two values. Own implementation to get rid of STL (VS6 problems)
template<class T>
NBL_FORCE_INLINE T abs(const T& a);
template<>
NBL_FORCE_INLINE vectorSIMDf abs<vectorSIMDf>(const vectorSIMDf& a);

//! returns 1 if a>0, 0 if a==0, -1 if a<0
template<class T>
NBL_FORCE_INLINE T sign(const T& a)
{
	auto isneg = a < T(0);
	using bool_type = std::remove_reference_t<decltype(isneg)>;
	T half_sign = core::mix<T,bool_type>(T(1),T(-1),isneg);
	auto iszero = a == T(0);
	return core::mix<T,bool_type>(half_sign,T(0),iszero);
}

//! round down
template<typename T>
NBL_FORCE_INLINE T floor(const T& x);
template<>
NBL_FORCE_INLINE float floor<float>(const float& x);
template<>
NBL_FORCE_INLINE double floor<double>(const double& x);
template<>
NBL_FORCE_INLINE vectorSIMDf floor<vectorSIMDf>(const vectorSIMDf& x);/*
template<typename T, typename I>
NBL_FORCE_INLINE I floor(const T& x)
{
	return I(core::floor<T>(x));
}*/

//! round towards zero
template<typename T>
NBL_FORCE_INLINE T trunc(const T& x);/*
template<typename T, typename I>
NBL_FORCE_INLINE I trunc(const T& x)
{
	return I(core::trunc<T>(x));
}*/

//! round to nearest integer
template<typename T>
NBL_FORCE_INLINE T round(const T& x)
{
	return core::floor<T>(x+T(0.5));
}
template<typename T, typename I>
NBL_FORCE_INLINE I round(const T& x)
{
	return I(core::round<T>(x));
}

//! No idea how this shit works, unimplemented
template<typename T>
NBL_FORCE_INLINE T roundEven(const T& x);
template<typename T, typename I>
NBL_FORCE_INLINE I roundEven(const T& x)
{
	return I(core::trunc<T>(x));
}

//! round up
template<typename T>
NBL_FORCE_INLINE T ceil(const T& x);
template<>
NBL_FORCE_INLINE float ceil<float>(const float& x);
template<>
NBL_FORCE_INLINE double ceil<double>(const double& x);
template<>
NBL_FORCE_INLINE vectorSIMDf ceil<vectorSIMDf>(const vectorSIMDf& x);/*
template<typename T, typename I>
NBL_FORCE_INLINE I ceil(const T& x)
{
	return I(core::ceil<T>(x));
}*/

//! 
template<typename T>
NBL_FORCE_INLINE T fract(const T& x)
{
	return x - core::floor<T>(x);
}

//! 
template<typename T, typename U=T>
NBL_FORCE_INLINE T mod(const T& x, const U& y)
{
	return core::fma<T>(-core::floor<T>(x / y),y,x);
}

//! 
template<typename T, typename U>
NBL_FORCE_INLINE T modf(const T& x, U& i)
{
	i = x; // floor
	return x - T(i);
}

template<class T>
NBL_FORCE_INLINE T min(const T& a, const T& b);
template<>
NBL_FORCE_INLINE vectorSIMDf min<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);
template<class T, typename U>
NBL_FORCE_INLINE T min(const T& a, const U& b)
{
	return core::min<T>(a,T(b));
}

template<class T, typename U=T>
NBL_FORCE_INLINE T min(const T& a, const U& b, const U& c)
{
	T vb = T(b);
	return core::min<T,T>(core::min<T,T>(a,vb), min<T,U>(vb,c));
}
/* don't remember what I made it for
template<typename... Args>
struct min_t
{
	inline auto operator()(Args&&... args)
	{
		return core::min<Args...>(std::forward<Args>(args)...)
	}
};
*/

template<class T>
NBL_FORCE_INLINE T max(const T& a, const T& b);
template<>
NBL_FORCE_INLINE vectorSIMDf max<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);
template<class T, typename U>
NBL_FORCE_INLINE T max(const T& a, const U& b)
{
	return core::max<T>(a, T(b));
}

template<class T, typename U=T>
NBL_FORCE_INLINE T max(const T& a, const U& b, const U& c)
{
	T vb = T(b);
	return core::max<T,T>(core::max<T,T>(a,vb),max<T,U>(vb,c));
}
/* don't remember what I made it for
template<typename... Args>
struct max_t
{
	inline auto operator()(Args&&... args)
	{
		return core::max<Args...>(std::forward<Args>(args)...)
	}
};
*/

//! clamps a value between low and high
template <class T, typename U=T>
NBL_FORCE_INLINE const T clamp(const T& value, const U& low, const U& high)
{
	return core::min<T,U>(core::max<T,U>(value, low), high);
}

//! alias for core::mix()
template<typename T, typename U=T>
NBL_FORCE_INLINE T lerp(const T& a, const T& b, const U& t)
{
	return core::mix<T,U>(a,b,t);
}


// TODO : step,smoothstep,isnan,isinf,floatBitsToInt,floatBitsToUint,intBitsToFloat,uintBitsToFloat,frexp,ldexp
// extra note, GCC breaks isfinite, isinf, isnan, isnormal, signbit in -ffast-math so need to implement ourselves
// TODO : packUnorm2x16, packSnorm2x16, packUnorm4x8, packSnorm4x8, unpackUnorm2x16, unpackSnorm2x16, unpackUnorm4x8, unpackSnorm4x8, packHalf2x16, unpackHalf2x16, packDouble2x32, unpackDouble2x32
// MOVE : faceforward, reflect, refract, any, all, not
template<typename T>
NBL_FORCE_INLINE T dot(const T& a, const T& b);
template<>
NBL_FORCE_INLINE vectorSIMDf dot<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);
template<>
NBL_FORCE_INLINE vectorSIMD_32<int32_t> dot<vectorSIMD_32<int32_t>>(const vectorSIMD_32<int32_t>& a, const vectorSIMD_32<int32_t>& b);
template<>
NBL_FORCE_INLINE vectorSIMD_32<uint32_t> dot<vectorSIMD_32<uint32_t>>(const vectorSIMD_32<uint32_t>& a, const vectorSIMD_32<uint32_t>& b);


template<typename T>
NBL_FORCE_INLINE T lengthsquared(const T& v)
{
	return core::dot<T>(v, v);
}
template<typename T>
NBL_FORCE_INLINE T length(const T& v)
{
	return core::sqrt<T>(lengthsquared<T>(v));
}

template<typename T>
NBL_FORCE_INLINE T distance(const T& a, const T& b)
{
	return core::length<T>(a-b);
}
template<typename T>
NBL_FORCE_INLINE T distancesquared(const T& a, const T& b)
{
	return core::lengthsquared<T>(a-b);
}

template<typename T>
NBL_FORCE_INLINE T cross(const T& a, const T& b);
template<>
NBL_FORCE_INLINE vectorSIMDf cross<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);

template<typename T>
NBL_FORCE_INLINE T normalize(const T& v)
{
	auto d = dot<T>(v, v);
#ifdef __NBL_FAST_MATH
	return v * core::inversesqrt<T>(d);
#else
	return v / core::sqrt<T>(d);
#endif
}

// TODO : matrixCompMult, outerProduct, inverse
template<typename T>
NBL_FORCE_INLINE T transpose(const T& m);
template<>
NBL_FORCE_INLINE matrix4SIMD transpose(const matrix4SIMD& m);




// Extras

template <typename T>
NBL_FORCE_INLINE constexpr std::enable_if_t<std::is_integral_v<T> && !std::is_signed_v<T>, T> bitfieldExtract(T value, int32_t offset, int32_t bits)
{
	constexpr T one = static_cast<T>(1);

	T retval = value;
	retval >>= offset;
	return retval & ((one<<bits) - one);
}
template <typename T>
NBL_FORCE_INLINE constexpr std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, T> bitfieldExtract(T value, int32_t offset, int32_t bits)
{
	constexpr T one = static_cast<T>(1);
	constexpr T all_set = static_cast<T>(~0ull);

	T retval = value;
	retval >>= offset;
	retval &= ((one<<bits) - one);
	if (retval & (one << (bits-1)))//sign extension
		retval |= (all_set << bits);

	return retval;
}

template <typename T>
NBL_FORCE_INLINE constexpr T bitfieldInsert(T base, T insert, int32_t offset, int32_t bits)
{
	constexpr T one = static_cast<T>(1);
	const T mask = (one << bits) - one;
	const T shifted_mask = mask << offset;

	insert &= mask;
	base &= (~shifted_mask);
	base |= (insert << offset);

	return base;
}

//! returns if a equals b, taking possible rounding errors into account
template<typename T>
NBL_FORCE_INLINE bool equals(const T& a, const T& b, const T& tolerance);
template<>
NBL_FORCE_INLINE bool equals<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& tolerance);
template<>
NBL_FORCE_INLINE bool equals<matrix4SIMD>(const matrix4SIMD& a, const matrix4SIMD& b, const matrix4SIMD& tolerance);
template<>
NBL_FORCE_INLINE bool equals<matrix3x4SIMD>(const matrix3x4SIMD& a, const matrix3x4SIMD& b, const matrix3x4SIMD& tolerance);


//! returns if a equals zero, taking rounding errors into account
template<typename T>
NBL_FORCE_INLINE auto iszero(const T& a, const T& tolerance)
{
	return core::abs<T>(a) <= tolerance;
}
template<typename T, typename U=T>
NBL_FORCE_INLINE auto iszero(const T& a, const U& tolerance = ROUNDING_ERROR<U>())
{
	return core::iszero(a,T(tolerance));
}


template<typename T>
NBL_FORCE_INLINE T sin(const T& a);


NBL_FORCE_INLINE float& intBitsToFloat(int32_t& _i) { return reinterpret_cast<float&>(_i); }
NBL_FORCE_INLINE float& uintBitsToFloat(uint32_t& _u) { return reinterpret_cast<float&>(_u); }
NBL_FORCE_INLINE int32_t& floatBitsToInt(float& _f) { return reinterpret_cast<int32_t&>(_f); }
NBL_FORCE_INLINE uint32_t& floatBitsToUint(float& _f) { return reinterpret_cast<uint32_t&>(_f); }
//rvalue ref parameters to ensure that functions returning a copy will be called for rvalues only (functions for lvalues returns same memory but with reinterpreted type)
NBL_FORCE_INLINE float intBitsToFloat(int32_t&& _i) { return reinterpret_cast<float&>(_i); }
NBL_FORCE_INLINE float uintBitsToFloat(uint32_t&& _u) { return reinterpret_cast<float&>(_u); }
NBL_FORCE_INLINE int32_t floatBitsToInt(float&& _f) { return reinterpret_cast<int32_t&>(_f); }
NBL_FORCE_INLINE uint32_t floatBitsToUint(float&& _f) { return reinterpret_cast<uint32_t&>(_f); }



// extras


template<typename T>
NBL_FORCE_INLINE T gcd(const T& a, const T& b);

template<typename T>
NBL_FORCE_INLINE T sinc(const T& x)
{
	// TODO: do a direct series/computation in the future
	return mix<T>(	sin<T>(x)/x,
					T(1.0)+x*x*(x*x*T(1.0/120.0)-T(1.0/6.0)),
					abs<T>(x)<T(0.0001)
				);
}
template<typename T>
NBL_FORCE_INLINE T d_sinc(const T& x)
{
	// TODO: do a direct series/computation in the future
	return mix<T>(	(cos<T>(x)-sin<T>(x)/x)/x,
					x*(x*x*T(4.0/120.0)-T(2.0/6.0)),
					abs<T>(x)<T(0.0001)
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

