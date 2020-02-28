// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_GLSL_FUNCTIONS_H_INCLUDED__
#define __IRR_GLSL_FUNCTIONS_H_INCLUDED__

#include <type_traits>
#include "irr/static_if.h"
#include "irr/type_traits.h"
#include "irr/core/math/floatutil.h"

namespace irr
{
namespace core
{

template<int components>
class vectorSIMDBool;
class vectorSIMDf;
class matrix4SIMD;
class matrix3x4SIMD;


template<typename T>
IRR_FORCE_INLINE T radians(const T& degrees)
{
	return degrees*PI<T>()/T(180);
}
template<typename T>
IRR_FORCE_INLINE T degrees(const T& radians)
{
	return radians*T(180)/PI<T>();
}
// TODO : sin,cos,tan,asin,acos,atan(y,x),atan(y_over_x),sinh,cosh,tanh,asinh,acosh,atanh,sincos

// TODO : pow,exp,log,exp2,log2
template<typename T>
IRR_FORCE_INLINE T sqrt(const T& x);
template<>
IRR_FORCE_INLINE float sqrt<float>(const float& x);
template<>
IRR_FORCE_INLINE double sqrt<double>(const double& x);
template<>
IRR_FORCE_INLINE vectorSIMDf sqrt<vectorSIMDf>(const vectorSIMDf& x);

template<typename T>
IRR_FORCE_INLINE T inversesqrt(const T& x);

template<typename T>
IRR_FORCE_INLINE T reciprocal(const T& x)
{
	return T(1.0)/x;
}

template<typename T>
IRR_FORCE_INLINE T reciprocal_approxim(const T& x);
template<>
IRR_FORCE_INLINE float reciprocal_approxim<float>(const float& x);
template<>
IRR_FORCE_INLINE vectorSIMDf reciprocal_approxim<vectorSIMDf>(const vectorSIMDf& x);

//! TODO : find some intrinsics
template<typename T>
IRR_FORCE_INLINE T fma(const T& a, const T& b, const T& c)
{
	return a * b + c;
}

//! returns linear interpolation of a and b with ratio t
//! only defined for floating point scalar and vector types
//! \return: a if t==0, b if t==1, and the linear interpolation else
template<typename T, typename U>
IRR_FORCE_INLINE T mix(const T & a, const T & b, const U & t)
{
	T retval;
	IRR_PSEUDO_IF_CONSTEXPR_BEGIN(irr::is_any_of<U,vectorSIMDBool<2>,vectorSIMDBool<4>,vectorSIMDBool<8>,vectorSIMDBool<16> >::value)
	{
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T,vectorSIMDf>::value)
			retval = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128((a&(~t)).getAsRegister()),_mm_castps_si128((b&t).getAsRegister())));
		IRR_PSEUDO_ELSE_CONSTEXPR
			retval = (a&(~t))|(b&t);
		IRR_PSEUDO_IF_CONSTEXPR_END;
	}
	IRR_PSEUDO_ELSE_CONSTEXPR
	{
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<U,bool>::value)
			retval = t ? b:a;
		IRR_PSEUDO_ELSE_CONSTEXPR
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(irr::is_any_of<T,matrix4SIMD,matrix3x4SIMD>::value)
			{
				for (uint32_t i=0u; i<T::VectorCount; i++)
				{
					IRR_PSEUDO_IF_CONSTEXPR_BEGIN(irr::is_any_of<U, matrix4SIMD, matrix3x4SIMD>::value)
						retval[i] = core::mix<vectorSIMDf, vectorSIMDf>(a.rows[i], b.rows[i], t.rows[i]);
					IRR_PSEUDO_ELSE_CONSTEXPR
						retval[i] = core::mix<vectorSIMDf, U>(a.rows[i], b.rows[i], t);
					IRR_PSEUDO_IF_CONSTEXPR_END;
				}
			}
			IRR_PSEUDO_ELSE_CONSTEXPR
				retval = core::fma<T>(b-a,t,a);
			IRR_PSEUDO_IF_CONSTEXPR_END;
		IRR_PSEUDO_IF_CONSTEXPR_END;
	}
	IRR_PSEUDO_IF_CONSTEXPR_END;
	return retval;
}

//! returns abs of two values. Own implementation to get rid of STL (VS6 problems)
template<class T>
IRR_FORCE_INLINE T abs(const T& a);
template<>
IRR_FORCE_INLINE vectorSIMDf abs<vectorSIMDf>(const vectorSIMDf& a);

//! returns 1 if a>0, 0 if a==0, -1 if a<0
template<class T>
IRR_FORCE_INLINE T sign(const T& a)
{
	auto isneg = a < T(0);
	using bool_type = typename std::remove_reference<decltype(isneg)>::type;
	T half_sign = core::mix<T,bool_type>(T(1),T(-1),isneg);
	auto iszero = a == T(0);
	return core::mix<T,bool_type>(half_sign,T(0),iszero);
}

//! round down
template<typename T>
IRR_FORCE_INLINE T floor(const T& x);
template<>
IRR_FORCE_INLINE float floor<float>(const float& x);
template<>
IRR_FORCE_INLINE double floor<double>(const double& x);
template<>
IRR_FORCE_INLINE vectorSIMDf floor<vectorSIMDf>(const vectorSIMDf& x);/*
template<typename T, typename I>
IRR_FORCE_INLINE I floor(const T& x)
{
	return I(core::floor<T>(x));
}*/

//! round towards zero
template<typename T>
IRR_FORCE_INLINE T trunc(const T& x);/*
template<typename T, typename I>
IRR_FORCE_INLINE I trunc(const T& x)
{
	return I(core::trunc<T>(x));
}*/

//! round to nearest integer
template<typename T>
IRR_FORCE_INLINE T round(const T& x)
{
	return core::floor<T>(x+T(0.5));
}
template<typename T, typename I>
IRR_FORCE_INLINE I round(const T& x)
{
	return I(core::round<T>(x));
}

//! No idea how this shit works, unimplemented
template<typename T>
IRR_FORCE_INLINE T roundEven(const T& x);
template<typename T, typename I>
IRR_FORCE_INLINE I roundEven(const T& x)
{
	return I(core::trunc<T>(x));
}

//! round up
template<typename T>
IRR_FORCE_INLINE T ceil(const T& x);
template<>
IRR_FORCE_INLINE float ceil<float>(const float& x);
template<>
IRR_FORCE_INLINE double ceil<double>(const double& x);
template<>
IRR_FORCE_INLINE vectorSIMDf ceil<vectorSIMDf>(const vectorSIMDf& x);/*
template<typename T, typename I>
IRR_FORCE_INLINE I ceil(const T& x)
{
	return I(core::ceil<T>(x));
}*/

//! 
template<typename T>
IRR_FORCE_INLINE T fract(const T& x)
{
	return x - core::floor<T>(x);
}

//! 
template<typename T, typename U=T>
IRR_FORCE_INLINE T mod(const T& x, const U& y)
{
	return core::fma<T>(-core::floor<T>(x / y),y,x);
}

//! 
template<typename T, typename U>
IRR_FORCE_INLINE T modf(const T& x, U& i)
{
	i = x; // floor
	return x - T(i);
}

template<class T>
IRR_FORCE_INLINE T min(const T& a, const T& b);
template<>
IRR_FORCE_INLINE vectorSIMDf min<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);
template<class T, typename U>
IRR_FORCE_INLINE T min(const T& a, const U& b)
{
	return core::min<T>(a,T(b));
}

template<class T, typename U=T>
IRR_FORCE_INLINE T min(const T& a, const U& b, const U& c)
{
	T vb = T(b);
	return core::min<T,T>(core::min<T,T>(a,vb), min<T,U>(vb,c));
}

template<class T>
IRR_FORCE_INLINE T max(const T& a, const T& b);
template<>
IRR_FORCE_INLINE vectorSIMDf max<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);
template<class T, typename U>
IRR_FORCE_INLINE T max(const T& a, const U& b)
{
	return core::max<T>(a, T(b));
}

template<class T, typename U=T>
IRR_FORCE_INLINE T max(const T& a, const U& b, const U& c)
{
	T vb = T(b);
	return core::max<T,T>(core::max<T,T>(a,vb),max<T,U>(vb,c));
}


//! clamps a value between low and high
template <class T, typename U=T>
IRR_FORCE_INLINE const T clamp(const T& value, const U& low, const U& high)
{
	return core::min<T,U>(core::max<T,U>(value, low), high);
}

//! alias for core::mix()
template<typename T, typename U=T>
IRR_FORCE_INLINE T lerp(const T& a, const T& b, const U& t)
{
	return core::mix<T,U>(a,b,t);
}


// TODO : step,smoothstep,isnan,isinf,floatBitsToInt,floatBitsToUint,intBitsToFloat,uintBitsToFloat,frexp,ldexp
// extra note, GCC breaks isfinite, isinf, isnan, isnormal, signbit in -ffast-math so need to implement ourselves
// TODO : packUnorm2x16, packSnorm2x16, packUnorm4x8, packSnorm4x8, unpackUnorm2x16, unpackSnorm2x16, unpackUnorm4x8, unpackSnorm4x8, packHalf2x16, unpackHalf2x16, packDouble2x32, unpackDouble2x32
// MOVE : faceforward, reflect, refract, any, all, not
template<typename T>
IRR_FORCE_INLINE T dot(const T& a, const T& b);
template<>
IRR_FORCE_INLINE vectorSIMDf dot<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);


template<typename T>
IRR_FORCE_INLINE T lengthsquared(const T& v)
{
	return core::dot<T>(v, v);
}
template<typename T>
IRR_FORCE_INLINE T length(const T& v)
{
	return core::sqrt<T>(lengthsquared<T>(v));
}

template<typename T>
IRR_FORCE_INLINE T distance(const T& a, const T& b)
{
	return core::length<T>(a-b);
}
template<typename T>
IRR_FORCE_INLINE T distancesquared(const T& a, const T& b)
{
	return core::lengthsquared<T>(a-b);
}

template<typename T>
IRR_FORCE_INLINE T cross(const T& a, const T& b);
template<>
IRR_FORCE_INLINE vectorSIMDf cross<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b);

template<typename T>
IRR_FORCE_INLINE T normalize(const T& v)
{
	auto d = dot<T>(v, v);
#ifdef __IRR_FAST_MATH
	return v * core::inversesqrt<T>(d);
#else
	return v / core::sqrt<T>(d);
#endif
}

// TODO : matrixCompMult, outerProduct, inverse
template<typename T>
IRR_FORCE_INLINE T transpose(const T& m);
template<>
IRR_FORCE_INLINE matrix4SIMD transpose(const matrix4SIMD& m);

template<typename T>
IRR_FORCE_INLINE float determinant(const T& m);
template<>
IRR_FORCE_INLINE float determinant(const matrix4SIMD& m);

// MAKE AIASES: lessThan, lessThanEqual, greaterThan, greaterThanEqual, equal, notEqual
// TODO : uaddCarry, usubBorrow, umulExtended, imulExtended, bitfieldExtract, bitfieldInsert, bitfieldReverse, bitCount

template<typename INT_TYPE>
IRR_FORCE_INLINE int32_t findLSB(INT_TYPE x);
template<>
IRR_FORCE_INLINE int32_t findLSB<uint32_t>(uint32_t x);
template<>
IRR_FORCE_INLINE int32_t findLSB<uint64_t>(uint64_t x);

template<typename INT_TYPE>
IRR_FORCE_INLINE int32_t findMSB(INT_TYPE x);
template<>
IRR_FORCE_INLINE int32_t findMSB<uint32_t>(uint32_t x);
template<>
IRR_FORCE_INLINE int32_t findMSB<uint64_t>(uint64_t x);

template<typename INT_TYPE>
IRR_FORCE_INLINE uint32_t bitCount(INT_TYPE x);
template<>
IRR_FORCE_INLINE uint32_t bitCount(uint32_t x);
template<>
IRR_FORCE_INLINE uint32_t bitCount(uint64_t x);
template<>
IRR_FORCE_INLINE uint32_t bitCount(int32_t x) {return core::bitCount(static_cast<const uint32_t&>(x));}
template<>
IRR_FORCE_INLINE uint32_t bitCount(int64_t x) {return core::bitCount(static_cast<const uint64_t&>(x));}

// Extras

//! returns if a equals b, taking possible rounding errors into account
template<typename T>
IRR_FORCE_INLINE bool equals(const T& a, const T& b, const T& tolerance);
template<>
IRR_FORCE_INLINE bool equals<vectorSIMDf>(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& tolerance);
template<>
IRR_FORCE_INLINE bool equals<matrix4SIMD>(const matrix4SIMD& a, const matrix4SIMD& b, const matrix4SIMD& tolerance);
template<>
IRR_FORCE_INLINE bool equals<matrix3x4SIMD>(const matrix3x4SIMD& a, const matrix3x4SIMD& b, const matrix3x4SIMD& tolerance);


//! returns if a equals zero, taking rounding errors into account
template<typename T>
IRR_FORCE_INLINE auto iszero(const T& a, const T& tolerance)
{
	return core::abs<T>(a) <= tolerance;
}
template<typename T, typename U=T>
IRR_FORCE_INLINE auto iszero(const T& a, const U& tolerance = ROUNDING_ERROR<U>())
{
	return core::iszero(a,T(tolerance));
}

} // end namespace core
} // end namespace irr

#endif

