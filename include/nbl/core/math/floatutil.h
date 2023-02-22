// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_FLOAT_UTIL_H_INCLUDED__
#define __NBL_CORE_FLOAT_UTIL_H_INCLUDED__


#include <float.h>
#include <stdint.h>
#include <cmath>
#include <algorithm>

#include "BuildConfigOptions.h"
#include "nbl/macros.h"


#ifndef FLT_MAX
#define FLT_MAX 3.402823466E+38F
#endif

#ifndef FLT_MIN
#define FLT_MIN 1.17549435e-38F
#endif

namespace nbl::core
{

class vectorSIMDf;
class matrix3x4SIMD;
class matrix4SIMD;

//! Rounding error constant often used when comparing values. (TODO: remove)
template<typename T>
NBL_FORCE_INLINE T ROUNDING_ERROR();
template<>
NBL_FORCE_INLINE float ROUNDING_ERROR<float>();
template<>
NBL_FORCE_INLINE double ROUNDING_ERROR<double>();
template<>
NBL_FORCE_INLINE vectorSIMDf ROUNDING_ERROR<vectorSIMDf>();
template<>
NBL_FORCE_INLINE matrix3x4SIMD ROUNDING_ERROR<matrix3x4SIMD>();
template<>
NBL_FORCE_INLINE matrix4SIMD ROUNDING_ERROR<matrix4SIMD>();

#ifdef PI // make sure we don't collide with a define
#undef PI
#endif
//! Constant for PI.
template<typename T>
NBL_FORCE_INLINE constexpr T PI()
{
	return T(3.14159265358979323846);
}

//! Constant for reciprocal of PI.
template<typename T>
NBL_FORCE_INLINE constexpr T RECIPROCAL_PI()
{
	return T(1.0) / PI<T>();
}

//! Constant for half of PI.
template<typename T>
NBL_FORCE_INLINE constexpr T HALF_PI()
{
	return PI<T>() * T(0.5);
}

namespace impl
{
	constexpr uint16_t NAN_U16		= 0x7FFFu;
    constexpr uint32_t NAN_U32      = 0x7FFFFFFFu;
	constexpr uint64_t NAN_U64		= 0x7fffffffFFFFFFFFull;
    constexpr uint32_t INFINITY_U32 = 0x7F800000u;
}

//! don't get rid of these, -ffast-math breaks inf and NaN handling on GCC
template<typename T>
T nan();

//TODO (?) we could have some core::half typedef or something..
template<>
NBL_FORCE_INLINE uint16_t nan<uint16_t>() { return impl::NAN_U16; }

template<>
NBL_FORCE_INLINE float nan<float>() { return reinterpret_cast<const float&>(impl::NAN_U32); }

template<>
NBL_FORCE_INLINE double nan<double>() { return reinterpret_cast<const double&>(impl::NAN_U64); }

template<typename T>
T infinity();

template<>
NBL_FORCE_INLINE float infinity<float>() {return reinterpret_cast<const float&>(impl::INFINITY_U32);}


template<typename T>
bool isnan(T val);

template<>
NBL_FORCE_INLINE bool isnan<double>(double val) 
{
    //all exponent bits set, at least one mantissa bit set
    return (reinterpret_cast<uint64_t&>(val)&0x7fffffffFFFFFFFFull) > 0x7ff0000000000000ull;
}

template<>
NBL_FORCE_INLINE bool isnan<float>(float val) 
{
    //all exponent bits set, at least one mantissa bit set
    return (reinterpret_cast<uint32_t&>(val)&0x7fffffffu) > 0x7f800000u;
}

template<>
NBL_FORCE_INLINE bool isnan<uint16_t>(uint16_t val)
{
	//all exponent bits set, at least one mantissa bit set
	return (val & 0x7fffu) > 0x7c00u;
}

template<typename T>
bool isinf(T val);

template<>
NBL_FORCE_INLINE bool isinf<float>(float val)
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

//! Integer representation of a floating-point value and the reverse.
#ifdef __NBL_FAST_MATH
NBL_FORCE_INLINE uint32_t& IR(float& x)
{
	return reinterpret_cast<uint32_t&>(x);
}
NBL_FORCE_INLINE const uint32_t& IR(const float& x)
{
	return reinterpret_cast<const uint32_t&>(x);
}
NBL_FORCE_INLINE float& FR(uint32_t& x)
{
	return reinterpret_cast<float&>(x);
}
NBL_FORCE_INLINE const float& FR(const uint32_t& x)
{
	return reinterpret_cast<const float&>(x);
}
NBL_FORCE_INLINE float& FR(int32_t& x)
{
	return reinterpret_cast<float&>(x);
}
NBL_FORCE_INLINE const float& FR(const int32_t& x)
{
	return reinterpret_cast<const float&>(x);
}
#else
// C++ full standard compat use memcpy
static_assert(sizeof(float)==sizeof(uint32_t));
NBL_FORCE_INLINE uint32_t IR(float x)
{
	uint32_t retval;
	memcpy(&retval,&x,sizeof(float));
	return retval;
}
NBL_FORCE_INLINE float FR(uint32_t x)
{
	float retval;
	memcpy(&retval, &x, sizeof(float));
	return retval;
}
NBL_FORCE_INLINE float FR(int32_t x)
{
	float retval;
	memcpy(&retval, &x, sizeof(float));
	return retval;
}
#endif

enum E_ROUNDING_DIRECTION : int8_t
{
	ERD_DOWN = -1,
	ERD_NEAREST = 0,
	ERD_UP = 1
};

//! We compare the difference in ULP's (spacing between floating-point numbers, aka ULP=1 means there exists no float between).
//\result true when numbers have a ULP <= maxUlpDiff AND have the same sign.
NBL_FORCE_INLINE bool equalsByUlp(float a, float b, int maxUlpDiff) //consider
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

struct rgb32f {
	float x, y, z;
};

/*
	RGB19E7
*/

[[maybe_unused]] constexpr uint32_t RGB19E7_EXP_BITS = 7u;
constexpr uint32_t RGB19E7_MANTISSA_BITS = 19u;
constexpr uint32_t RGB19E7_EXP_BIAS = 63u;
constexpr uint32_t RGB19E7_MAX_VALID_BIASED_EXP = 127u;
constexpr uint32_t MAX_RGB19E7_EXP = RGB19E7_MAX_VALID_BIASED_EXP - RGB19E7_EXP_BIAS;
constexpr uint32_t RGB19E7_MANTISSA_VALUES = 1u<<RGB19E7_MANTISSA_BITS;
constexpr uint32_t MAX_RGB19E7_MANTISSA = RGB19E7_MANTISSA_VALUES-1u;
constexpr float MAX_RGB19E7 = static_cast<float>(MAX_RGB19E7_MANTISSA)/RGB19E7_MANTISSA_VALUES * (1LL<<(MAX_RGB19E7_EXP-32)) * (1LL<<32);
[[maybe_unused]] constexpr float EPSILON_RGB19E7 = (1.f/RGB19E7_MANTISSA_VALUES) / (1LL<<(RGB19E7_EXP_BIAS-32)) / (1LL<<32);
// TODO: @Crisspl lets make the `rgb18e7s3` format and refactor some of this shared exponent magic
inline uint64_t rgb32f_to_rgb19e7(const float _rgb[3])
{
	union rgb19e7 {
		uint64_t u64;
		struct field {
			uint64_t r : 19;
			uint64_t g : 19;
			uint64_t b : 19;
			uint64_t e : 7;
		} field;
	};

	auto clamp_rgb19e7 = [=](float x) -> float {
		return std::max(0.f, std::min(x, MAX_RGB19E7));
	};

	const float r = clamp_rgb19e7(_rgb[0]);
	const float g = clamp_rgb19e7(_rgb[1]);
	const float b = clamp_rgb19e7(_rgb[2]);

	auto f32_exp = [](float x) -> int32_t { return ((reinterpret_cast<int32_t&>(x)>>23) & 0xff) - 127; };

	const float maxrgb = std::max({r,g,b});
	int32_t exp_shared = std::max(-static_cast<int32_t>(RGB19E7_EXP_BIAS)-1, f32_exp(maxrgb)) + 1 + RGB19E7_EXP_BIAS;
	assert(exp_shared <= static_cast<int32_t>(RGB19E7_MAX_VALID_BIASED_EXP));
	assert(exp_shared >= 0);

	double denom = std::pow(2.0, static_cast<int32_t>(exp_shared-RGB19E7_EXP_BIAS-RGB19E7_MANTISSA_BITS)); // TODO: use fast exp2 and compute reciprocal instead, also use floats not doubles

	const uint32_t maxm = static_cast<uint32_t>(maxrgb/denom+0.5);
	if (maxm == MAX_RGB19E7_MANTISSA+1u)
	{
		denom *= 2.0;
		++exp_shared;
		assert(exp_shared <= static_cast<int32_t>(RGB19E7_MAX_VALID_BIASED_EXP));
	}
	else
	{
		assert(maxm <= MAX_RGB19E7_MANTISSA);
	}

	int32_t rm = r/denom + 0.5;
	int32_t gm = g/denom + 0.5;
	int32_t bm = b/denom + 0.5;

	assert(rm <= static_cast<int32_t>(MAX_RGB19E7_MANTISSA));
	assert(gm <= static_cast<int32_t>(MAX_RGB19E7_MANTISSA));
	assert(bm <= static_cast<int32_t>(MAX_RGB19E7_MANTISSA));
	assert(rm >= 0);
	assert(gm >= 0);
	assert(bm >= 0);

	rgb19e7 retval;
	retval.field.r = rm;
	retval.field.g = gm;
	retval.field.b = bm;
	retval.field.e = exp_shared;

	return retval.u64;
}
inline uint64_t rgb32f_to_rgb19e7(float r, float g, float b)
{
	const float rgb[3]{ r,g,b };

	return rgb32f_to_rgb19e7(rgb);
}
inline rgb32f rgb19e7_to_rgb32f(uint64_t _rgb19e7)
{
	union rgb19e7 {
		uint64_t u64;
		struct field {
			uint64_t r : 19;
			uint64_t g : 19;
			uint64_t b : 19;
			uint64_t e : 7;
		} field;
	};
	rgb19e7 u;
	u.u64 = _rgb19e7;
	int32_t exp = u.field.e - RGB19E7_EXP_BIAS - RGB19E7_MANTISSA_BITS;
	float scale = static_cast<float>(std::exp2(exp));

	rgb32f r;
	r.x = u.field.r * scale;
	r.y = u.field.g * scale;
	r.z = u.field.b * scale;

	return r;
}

uint32_t& floatBitsToUint(float& _f);
uint32_t floatBitsToUint(float&& _f);

/*
	RGB18E7S3
*/

[[maybe_unused]] constexpr uint32_t RGB18E7S3_EXP_BITS = 7u;
constexpr uint32_t RGB18E7S3_MANTISSA_BITS = 18u;
constexpr uint32_t RGB18E7S3_EXP_BIAS = 63;	//! 2^(RGB18E7S3_EXP_BITS - 1) - 1
constexpr uint32_t RGB18E7S3_MAX_VALID_BIASED_EXP = 127u; //! 2^RGB18E7S3_EXP_BITS - 1

constexpr uint32_t MAX_RGB18E7S3_EXP = RGB18E7S3_MAX_VALID_BIASED_EXP - RGB18E7S3_EXP_BIAS;
constexpr uint32_t RGB18E7S3_MANTISSA_VALUES = 1u << RGB18E7S3_MANTISSA_BITS;
constexpr uint32_t MAX_RGB18E7S3_MANTISSA = RGB18E7S3_MANTISSA_VALUES - 1u;
constexpr float MAX_RGB18E7S3 = static_cast<float>(MAX_RGB18E7S3_MANTISSA) / RGB18E7S3_MANTISSA_VALUES * (1LL << (MAX_RGB18E7S3_EXP - 32)) * (1LL << 32);
[[maybe_unused]] constexpr float EPSILON_RGB18E7S3 = (1.f / RGB18E7S3_MANTISSA_VALUES) / (1LL << (RGB18E7S3_EXP_BIAS - 32)) / (1LL << 32);

template<E_ROUNDING_DIRECTION rounding=ERD_NEAREST>
inline uint64_t rgb32f_to_rgb18e7s3(const float _rgb[3])
{
	union rgb18e7s3 {
		uint64_t u64;
		struct field {
			uint64_t r : 18;
			uint64_t g : 18;
			uint64_t b : 18;
			uint64_t e : 7;
			uint64_t s : 3;
		} field; // TODO: @AnastazIuk YOU ABSOLUTELY CANNOT RELY ON BITFIELD DATA LAYOUT ACROSS COMPILERS!
	};

	auto clamp_rgb18e7s3 = [=](float x) -> float {
		return std::max(0.f, std::min(x, MAX_RGB18E7S3));
	};

	float r = clamp_rgb18e7s3(abs(_rgb[0]));
	float g = clamp_rgb18e7s3(abs(_rgb[1]));
	float b = clamp_rgb18e7s3(abs(_rgb[2]));

	auto f32_exp = [](float x) -> int32_t { return ((reinterpret_cast<int32_t&>(x) >> 23) & 0xff) - 127; };

	const float maxrgb = std::max({ r,g,b });
	int32_t exp_shared = std::max(-static_cast<int32_t>(RGB18E7S3_EXP_BIAS) - 1, f32_exp(maxrgb)) + 1 + RGB18E7S3_EXP_BIAS;
	assert(exp_shared <= static_cast<int32_t>(RGB18E7S3_MAX_VALID_BIASED_EXP));
	assert(exp_shared >= 0);

	double denom = std::pow(2.0, static_cast<int32_t>(exp_shared - RGB18E7S3_EXP_BIAS - RGB18E7S3_MANTISSA_BITS)); // TODO: use fast exp2 and compute reciprocal instead, also use floats not doubles

	const uint32_t maxm = static_cast<uint32_t>(maxrgb / denom + 0.5);
	if (maxm == MAX_RGB18E7S3_MANTISSA + 1u)
	{
		denom *= 2.0;
		++exp_shared;
		assert(exp_shared <= static_cast<int32_t>(RGB18E7S3_MAX_VALID_BIASED_EXP));
	}
	else
		assert(maxm <= MAX_RGB18E7S3_MANTISSA);
	
	r /= denom;
	g /= denom;
	b /= denom;

	const uint32_t signR = floatBitsToUint(float(_rgb[0]))&0x80000000u;
	const uint32_t signG = floatBitsToUint(float(_rgb[1]))&0x80000000u;
	const uint32_t signB = floatBitsToUint(float(_rgb[2]))&0x80000000u;

	int32_t rm,gm,bm;
	if constexpr(rounding==ERD_NEAREST)
	{
		const float absNear = 0.5f;
		rm = r + absNear;
		gm = g + absNear;
		bm = b + absNear;
	}
	else
	{
		const float absDown = 0.f;
		const float absUp = FR(0x3f7fffffu); // uintBitsToFloat
		if constexpr(rounding==ERD_DOWN)
		{
			rm = r + (signR ? absUp:absDown);
			gm = g + (signG ? absUp:absDown);
			bm = b + (signB ? absUp:absDown);
		}
		else
		{
			rm = r + (signR ? absDown:absUp);
			gm = g + (signG ? absDown:absUp);
			bm = b + (signB ? absDown:absUp);
		}
	}

	assert(rm <= static_cast<int32_t>(MAX_RGB18E7S3_MANTISSA));
	assert(gm <= static_cast<int32_t>(MAX_RGB18E7S3_MANTISSA));
	assert(bm <= static_cast<int32_t>(MAX_RGB18E7S3_MANTISSA));
	assert(rm >= 0);
	assert(gm >= 0);
	assert(bm >= 0);
	
	const auto signMask = static_cast<uint8_t>((signR>>31u)|(signG>>30u)|(signB>>29u));
	rgb18e7s3 returnValue;
	returnValue.field.r = rm;
	returnValue.field.g = gm;
	returnValue.field.b = bm;
	returnValue.field.e = exp_shared;
	returnValue.field.s = signMask;

	return returnValue.u64;
}
inline uint64_t rgb32f_to_rgb18e7s3(float r, float g, float b)
{
	const float rgb[3]{ r,g,b };

	return rgb32f_to_rgb18e7s3(rgb);
}

inline rgb32f rgb18e7s3_to_rgb32f(uint64_t _rgb18e7s3)
{
	union rgb18e7s3 {
		uint64_t u64;
		struct field {
			uint64_t r : 18;
			uint64_t g : 18;
			uint64_t b : 18;
			uint64_t e : 7;
			uint64_t s : 3;
		} field;
	};

	rgb18e7s3 u;
	u.u64 = _rgb18e7s3;
	int32_t exp = u.field.e - RGB18E7S3_EXP_BIAS - RGB18E7S3_MANTISSA_BITS;
	float scale = static_cast<float>(std::exp2(exp));
	const uint8_t signMask = u.field.s;

	const bool isEncodedRNegative = signMask & (1u << 0u);
	const bool isEncodedGNegative = signMask & (1u << 1u);
	const bool isEncodedBNegative = signMask & (1u << 2u);

	rgb32f rgb;
	rgb.x = u.field.r * scale;
	if (isEncodedRNegative)
		floatBitsToUint(rgb.x) ^= 0x80000000u;

	rgb.y = u.field.g * scale;
	if (isEncodedGNegative)
		floatBitsToUint(rgb.y) ^= 0x80000000u;

	rgb.z = u.field.b * scale;
	if (isEncodedBNegative)
		floatBitsToUint(rgb.z) ^= 0x80000000u;

	return rgb;
}

NBL_FORCE_INLINE float nextafter32(float x, float y)
{
	return std::nextafterf(x, y);
}
NBL_FORCE_INLINE double nextafter64(double x, double y)
{
	return std::nextafter(x, y);
}
inline uint16_t nextafter16(uint16_t x, uint16_t y)
{
	if (x==y)
		return y;

	if (isnan(x) || isnan(y))
		return impl::NAN_U16;

	uint16_t ax = x & ((1u<<15) - 1u);
	uint16_t ay = y & ((1u<<15) - 1u);
	if (ax == 0u) {
		if (ay == 0u)
			return y;
		x = (y & 1u<<15) | 1u;
	}
	else if (ax>ay || ((x ^ y) & 1u<<15))
		--x;
	else
		++x;

	return x;
}

}

#endif
