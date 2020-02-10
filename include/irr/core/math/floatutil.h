// Copyright (C) 2019 DevSH Graphics Programming Sp. z O.O.
// This file is part of the "IrrlichtBaW".
// For conditions of distribution and use, see LICENSE.md

#ifndef __IRR_FLOAT_UTIL_H_INCLUDED__
#define __IRR_FLOAT_UTIL_H_INCLUDED__


#include <float.h>
#include <stdint.h>

#include "irr/macros.h"


#ifndef FLT_MAX
#define FLT_MAX 3.402823466E+38F
#endif

#ifndef FLT_MIN
#define FLT_MIN 1.17549435e-38F
#endif

namespace irr
{
namespace core
{

class vectorSIMDf;
class matrix3x4SIMD;
class matrix4SIMD;

//! Rounding error constant often used when comparing values.
template<typename T>
IRR_FORCE_INLINE T ROUNDING_ERROR();
template<>
IRR_FORCE_INLINE float ROUNDING_ERROR<float>();
template<>
IRR_FORCE_INLINE double ROUNDING_ERROR<double>();
template<>
IRR_FORCE_INLINE vectorSIMDf ROUNDING_ERROR<vectorSIMDf>();
template<>
IRR_FORCE_INLINE matrix3x4SIMD ROUNDING_ERROR<matrix3x4SIMD>();
template<>
IRR_FORCE_INLINE matrix4SIMD ROUNDING_ERROR<matrix4SIMD>();

#ifdef PI // make sure we don't collide with a define
#undef PI
#endif
//! Constant for PI.
template<typename T>
IRR_FORCE_INLINE T PI()
{
	return T(3.14159265358979323846);
}

//! Constant for reciprocal of PI.
template<typename T>
IRR_FORCE_INLINE T RECIPROCAL_PI()
{
	return T(1.0) / PI<T>();
}

//! Constant for half of PI.
template<typename T>
IRR_FORCE_INLINE T HALF_PI()
{
	return PI<T>() * T(0.5);
}

namespace impl
{
    constexpr uint32_t NAN_U32      = 0x7FFFFFFFu;
    constexpr uint32_t INFINITY_U32 = 0x7F800000u;
}

//! don't get rid of these, -ffast-math breaks inf and NaN handling on GCC
template<typename T>
T nan();

template<>
IRR_FORCE_INLINE float nan<float>() {return reinterpret_cast<const float&>(impl::NAN_U32);}


template<typename T>
T infinity();

template<>
IRR_FORCE_INLINE float infinity<float>() {return reinterpret_cast<const float&>(impl::INFINITY_U32);}


template<typename T>
bool isnan(T val);

template<>
IRR_FORCE_INLINE bool isnan<float>(float val) 
{
    //all exponent bits set, at least one mantissa bit set
    return (reinterpret_cast<uint32_t&>(val)&0x7fffffffu) > 0x7f800000u;
}

template<typename T>
bool isinf(T val);

template<>
IRR_FORCE_INLINE bool isinf<float>(float val)
{
    //all exponent bits set, none mantissa bit set
    return (reinterpret_cast<uint32_t&>(val)&0x7fffffffu) == 0x7f800000u;
}

union FloatIntUnion32
{
	FloatIntUnion32(float f1 = 0.0f) : f(f1) {}
	// Portable sign-extraction
	bool sign() const { return (i >> 31) != 0; }

	int32_t i;
	float f;
};

//! code is taken from IceFPU
//! Integer representation of a floating-point value and the reverse.
#ifdef __IRR_FAST_MATH
#define IR(x)                           ((uint32_t&)(x))
#define FR(x)                           ((float&)(x))
#else
IRR_FORCE_INLINE uint32_t IR(float x) { inttofloat tmp; tmp.f = x; return tmp.u; }
IRR_FORCE_INLINE float FR(uint32_t x) { inttofloat tmp; tmp.u = x; return tmp.f; }
IRR_FORCE_INLINE float FR(int32_t x) { inttofloat tmp; tmp.s = x; return tmp.f; }
#endif

//! We compare the difference in ULP's (spacing between floating-point numbers, aka ULP=1 means there exists no float between).
//\result true when numbers have a ULP <= maxUlpDiff AND have the same sign.
IRR_FORCE_INLINE bool equalsByUlp(float a, float b, int maxUlpDiff) //consider
{
	// Based on the ideas and code from Bruce Dawson on
	// http://www.altdevblogaday.com/2012/02/22/comparing-floating-point-numbers-2012-edition/
	// When floats are interpreted as integers the two nearest possible float numbers differ just
	// by one integer number. Also works the other way round, an integer of 1 interpreted as float
	// is for example the smallest possible float number.

	FloatIntUnion32 fa(a);
	FloatIntUnion32 fb(b);

	// Different signs, we could maybe get difference to 0, but so close to 0 using epsilons is better.
	if (fa.sign() != fb.sign())
	{
		// Check for equality to make sure +0==-0
		if (fa.i == fb.i)
			return true;
		return false;
	}

	// Find the difference in ULPs.
	int ulpsDiff = (fa.i - fb.i);
	if (ulpsDiff <= maxUlpDiff || -ulpsDiff >= -maxUlpDiff)
		return true;

	return false;
}


/** Floating point number in unsigned 11-bit floating point format shall be on eleven youngest bits of _fp. Used for ECT_UNSIGNED_INT_10F_11F_11F_REV attribute type (OpenGL's GL_UNSIGNED_INT_10F_11F_11F_REV). */
inline float unpack11bitFloat(uint32_t _fp)
{
	const uint32_t mask = 0x7ffu;
	const uint32_t mantissaMask = 0x3fu;
	const uint32_t expMask = 0x7c0u;

	_fp &= mask;

	if (!_fp)
		return 0.f;

	const uint32_t mant = _fp & mantissaMask;
	const uint32_t exp = (_fp & expMask) >> 6;
	if (exp < 31 && mant)
	{
		float f32 = 0.f;
		uint32_t& if32 = *((uint32_t*)& f32);

		if32 |= (mant << (23 - 6));
		if32 |= ((exp + (127 - 15)) << 23);
		return f32;
	}
	else if (exp == 31 && !mant)
		return core::infinity<float>();
	else if (exp == 31 && mant)
		return core::nan<float>();

	return -1.f;
}

/** Conversion to 11bit unsigned floating point format. Result is located on 11 youngest bits of returned variable. Used for ECT_UNSIGNED_INT_10F_11F_11F_REV attribute type (OpenGL's GL_UNSIGNED_INT_10F_11F_11F_REV). */
inline uint32_t to11bitFloat(float _f32)
{
	const uint32_t& f32 = *((uint32_t*)& _f32);

	if (f32 & 0x80000000u) // negative numbers converts to 0 (represented by all zeroes in 11bit format)
		return 0;

	const uint32_t f11MantissaMask = 0x3fu;
	const uint32_t f11ExpMask = 0x1fu << 6;

	const int32_t exp = ((f32 >> 23) & 0xffu) - 127;
	const uint32_t mantissa = f32 & 0x7fffffu;

	uint32_t f11 = 0u;
	if (exp == 128) // inf / NaN
	{
		f11 = f11ExpMask;
		if (mantissa)
			f11 |= (mantissa & f11MantissaMask);
	}
	else if (exp > 15) // overflow converts to infinity
		f11 = f11ExpMask;
	else if (exp > -15)
	{
		const int32_t e = exp + 15;
		const uint32_t m = mantissa >> (23 - 6);
		f11 = (e << 6) | m;
	}

	return f11;
}

/** Floating point number in unsigned 10-bit floating point format shall be on ten youngest bits of _fp. Used for ECT_UNSIGNED_INT_10F_11F_11F_REV attribute type (OpenGL's GL_UNSIGNED_INT_10F_11F_11F_REV). */
inline float unpack10bitFloat(uint32_t _fp)
{
	const uint32_t mask = 0x3ffu;
	const uint32_t mantissaMask = 0x1fu;
	const uint32_t expMask = 0x3e0u;

	_fp &= mask;

	if (!_fp)
		return 0.f;

	const uint32_t mant = _fp & mantissaMask;
	const uint32_t exp = (_fp & expMask) >> 5;
	if (exp < 31)
	{
		float f32 = 0.f;
		uint32_t& if32 = *((uint32_t*)& f32);

		if32 |= (mant << (23 - 5));
		if32 |= ((exp + (127 - 15)) << 23);
		return f32;
	}
	else if (exp == 31 && !mant)
		return core::infinity<float>();
	else if (exp == 31 && mant)
		return core::nan<float>();
	return -1.f;
}

/** Conversion to 10bit unsigned floating point format. Result is located on 10 youngest bits of returned variable. Used for ECT_UNSIGNED_INT_10F_11F_11F_REV attribute type (OpenGL's GL_UNSIGNED_INT_10F_11F_11F_REV). */
inline uint32_t to10bitFloat(float _f32)
{
	const uint32_t& f32 = *((uint32_t*)& _f32);

	if (f32 & 0x80000000u) // negative numbers converts to 0 (represented by all zeroes in 10bit format)
		return 0;

	const uint32_t f10MantissaMask = 0x1fu;
	const uint32_t f10ExpMask = 0x1fu << 5;

	const int32_t exp = ((f32 >> 23) & 0xffu) - 127;
	const uint32_t mantissa = f32 & 0x7fffffu;

	uint32_t f10 = 0u;
	if (exp == 128) // inf / NaN
	{
		f10 = f10ExpMask;
		if (mantissa)
			f10 |= (mantissa & f10MantissaMask);
	}
	else if (exp > 15) // overflow, converts to infinity
		f10 = f10ExpMask;
	else if (exp > -15)
	{
		const int32_t e = exp + 15;
		const uint32_t m = mantissa >> (23 - 5);
		f10 = (e << 5) | m;
	}

	return f10;
}

//! Utility class used for IEEE754 float32 <-> float16 conversions
/** By Phernost; taken from https://stackoverflow.com/a/3542975/5538150 */
class Float16Compressor
{
		union Bits
		{
			float f;
			int32_t si;
			uint32_t ui;
		};

		static int const shift = 13;
		static int const shiftSign = 16;

		static int32_t const infN = 0x7F800000; // flt32 infinity
		static int32_t const maxN = 0x477FE000; // max flt16 normal as a flt32
		static int32_t const minN = 0x38800000; // min flt16 normal as a flt32
		static int32_t const signN = 0x80000000; // flt32 sign bit

		static int32_t const infC = infN >> shift;
		static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
		static int32_t const maxC = maxN >> shift;
		static int32_t const minC = minN >> shift;
		static int32_t const signC = signN >> shiftSign; // flt16 sign bit

		static int32_t const mulN = 0x52000000; // (1 << 23) / minN
		static int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

		static int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
		static int32_t const norC = 0x00400; // min flt32 normal down shifted

		static int32_t const maxD = infC - maxC - 1;
		static int32_t const minD = minC - subC - 1;

		Float16Compressor() = delete;

	public:
		//! float32 -> float16
		static inline uint16_t compress(float value)
		{
			Bits v, s;
			v.f = value;
			uint32_t sign = v.si & signN;
			v.si ^= sign;
			sign >>= shiftSign; // logical shift
			s.si = mulN;
			s.si = static_cast<int32_t>(s.f * v.f); // correct subnormals
			v.si ^= (s.si ^ v.si) & -((int32_t)(minN > v.si));
			v.si ^= (infN ^ v.si) & -((int32_t)(infN > v.si) & (int32_t)(v.si > maxN));
			v.si ^= (nanN ^ v.si) & -((int32_t)(nanN > v.si) & (int32_t)(v.si > infN));
			v.ui >>= shift; // logical shift
			v.si ^= ((v.si - maxD) ^ v.si) & -((int32_t)(v.si > maxC));
			v.si ^= ((v.si - minD) ^ v.si) & -((int32_t)(v.si > subC));
			return v.ui | sign;
		}

		//! float16 -> float32
		static inline float decompress(uint16_t value)
		{
			Bits v;
			v.ui = value;
			int32_t sign = v.si & signC;
			v.si ^= sign;
			sign <<= shiftSign;
			v.si ^= ((v.si + minD) ^ v.si) & -(int32_t)(v.si > subC);
			v.si ^= ((v.si + maxD) ^ v.si) & -(int32_t)(v.si > maxC);
			Bits s;
			s.si = mulC;
			s.f *= v.si;
			int32_t mask = -((int32_t)(norC > v.si));
			v.si <<= shift;
			v.si ^= (s.si ^ v.si) & mask;
			v.si |= sign;
			return v.f;
		}
};

}
}

#endif
