// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MATH_H_INCLUDED__
#define __IRR_MATH_H_INCLUDED__

#include "IrrCompileConfig.h"

#include <stdint.h>
#include <math.h>
#include <float.h>
#include <stdlib.h> // for abs() etc.
#include <limits.h> // For INT_MAX / UINT_MAX
#include <type_traits>
#ifdef _MSC_VER
    #include <intrin.h>
#endif

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

	//! Rounding error constant often used when comparing float values.

	const int32_t ROUNDING_ERROR_S32 = 0;
	const int64_t ROUNDING_ERROR_S64 = 0;

	const float ROUNDING_ERROR_f32 = 0.000001f;
	const double ROUNDING_ERROR_f64 = 0.00000001;

#ifdef PI // make sure we don't collide with a define
#undef PI
#endif
	//! Constant for PI.
	const float PI		= 3.14159265359f;

	//! Constant for reciprocal of PI.
	const float RECIPROCAL_PI	= 1.0f/PI;

	//! Constant for half of PI.
	const float HALF_PI	= PI/2.0f;

#ifdef PI64 // make sure we don't collide with a define
#undef PI64
#endif
	//! Constant for 64bit PI.
	const double PI64		= 3.1415926535897932384626433832795028841971693993751;

	//! Constant for 64bit reciprocal of PI.
	const double RECIPROCAL_PI64 = 1.0/PI64;

	//! 32bit Constant for converting from degrees to radians
	const float DEGTORAD = PI / 180.0f;

	//! 32bit constant for converting from radians to degrees (formally known as GRAD_PI)
	const float RADTODEG   = 180.0f / PI;

	//! 64bit constant for converting from degrees to radians (formally known as GRAD_PI2)
	const double DEGTORAD64 = PI64 / 180.0;

	//! 64bit constant for converting from radians to degrees
	const double RADTODEG64 = 180.0 / PI64;

	//! Utility function to convert a radian value to degrees
	/** Provided as it can be clearer to write radToDeg(X) than RADTODEG * X
	\param radians	The radians value to convert to degrees.
	*/
	IRR_FORCE_INLINE float radToDeg(float radians)
	{
		return RADTODEG * radians;
	}

	//! Utility function to convert a radian value to degrees
	/** Provided as it can be clearer to write radToDeg(X) than RADTODEG * X
	\param radians	The radians value to convert to degrees.
	*/
	IRR_FORCE_INLINE double radToDeg(double radians)
	{
		return RADTODEG64 * radians;
	}

	//! Utility function to convert a degrees value to radians
	/** Provided as it can be clearer to write degToRad(X) than DEGTORAD * X
	\param degrees	The degrees value to convert to radians.
	*/
	IRR_FORCE_INLINE float degToRad(float degrees)
	{
		return DEGTORAD * degrees;
	}

	//! Utility function to convert a degrees value to radians
	/** Provided as it can be clearer to write degToRad(X) than DEGTORAD * X
	\param degrees	The degrees value to convert to radians.
	*/
	IRR_FORCE_INLINE double degToRad(double degrees)
	{
		return DEGTORAD64 * degrees;
	}
//depr
	//! returns minimum of two values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	IRR_FORCE_INLINE T min_(const T& a, const T& b)
	{
		return a < b ? a : b;
	}

	//! returns minimum of three values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	IRR_FORCE_INLINE T min_(const T& a, const T& b, const T& c)
	{
		return a < b ? min_(a, c) : min_(b, c);
	}

	//! returns maximum of two values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	IRR_FORCE_INLINE T max_(const T& a, const T& b)
	{
		return a < b ? b : a;
	}

	//! returns maximum of three values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	IRR_FORCE_INLINE T max_(const T& a, const T& b, const T& c)
	{
		return a < b ? max_(b, c) : max_(a, c);
	}

	//! returns abs of two values. Own implementation to get rid of STL (VS6 problems)
	template<class T>
	IRR_FORCE_INLINE T abs_(const T& a)
	{
		return a < (T)0 ? -a : a;
	}
//end depr
	//! returns linear interpolation of a and b with ratio t
	//! \return: a if t==0, b if t==1, and the linear interpolation else
	template<class T>
	IRR_FORCE_INLINE T mix(const T& a, const T& b, const float t)
	{
		return a + (b-a)*t;
	}

	//! returns linear interpolation of a and b with ratio t
	//! \return: a if t==0, b if t==1, and the linear interpolation else
	template<class T>
	IRR_FORCE_INLINE T lerp(const T& a, const T& b, const float t)
	{
		return mix(a,b,t);
	}

	//! clamps a value between low and high
	template <class T>
	IRR_FORCE_INLINE const T clamp(const T& value, const T& low, const T& high)
	{
		return min_ (max_(value,low), high);
	}

	//! returns if a equals b, taking possible rounding errors into account
	IRR_FORCE_INLINE bool equals(const double a, const double b, const double tolerance = ROUNDING_ERROR_f64) //consider
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals b, taking possible rounding errors into account
	IRR_FORCE_INLINE bool equals(const float a, const float b, const float tolerance = ROUNDING_ERROR_f32) //consider
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	union FloatIntUnion32 //todo: rename
	{
		FloatIntUnion32(float f1 = 0.0f) : f(f1) {}
		// Portable sign-extraction
		bool sign() const { return (i >> 31) != 0; }

		int32_t i;
		float f;
	};

	//! We compare the difference in ULP's (spacing between floating-point numbers, aka ULP=1 means there exists no float between).
	//\result true when numbers have a ULP <= maxUlpDiff AND have the same sign.
	inline bool equalsByUlp(float a, float b, int maxUlpDiff) //consider
	{
		// Based on the ideas and code from Bruce Dawson on
		// http://www.altdevblogaday.com/2012/02/22/comparing-floating-point-numbers-2012-edition/
		// When floats are interpreted as integers the two nearest possible float numbers differ just
		// by one integer number. Also works the other way round, an integer of 1 interpreted as float
		// is for example the smallest possible float number.

		FloatIntUnion32 fa(a);
		FloatIntUnion32 fb(b);

		// Different signs, we could maybe get difference to 0, but so close to 0 using epsilons is better.
		if ( fa.sign() != fb.sign() )
		{
			// Check for equality to make sure +0==-0
			if (fa.i == fb.i)
				return true;
			return false;
		}

		// Find the difference in ULPs.
		int ulpsDiff = abs_(fa.i- fb.i);
		if (ulpsDiff <= maxUlpDiff)
			return true;

		return false;
	}

#if 0
	//! returns if a equals b, not using any rounding tolerance
	IRR_FORCE_INLINE bool equals(const int32_t a, const int32_t b) //consider
	{
		return (a == b);
	}

	//! returns if a equals b, not using any rounding tolerance
	IRR_FORCE_INLINE bool equals(const uint32_t a, const uint32_t b) //consider
	{
		return (a == b);
	}
#endif
	//! returns if a equals b, taking an explicit rounding tolerance into account
	IRR_FORCE_INLINE bool equals(const int32_t a, const int32_t b, const int32_t tolerance = ROUNDING_ERROR_S32) //consider
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals b, taking an explicit rounding tolerance into account
	IRR_FORCE_INLINE bool equals(const uint32_t a, const uint32_t b, const int32_t tolerance = ROUNDING_ERROR_S32) //consider
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals b, taking an explicit rounding tolerance into account
	IRR_FORCE_INLINE bool equals(const int64_t a, const int64_t b, const int64_t tolerance = ROUNDING_ERROR_S64) //consider
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals zero, taking rounding errors into account
	IRR_FORCE_INLINE bool iszero(const double a, const double tolerance = ROUNDING_ERROR_f64) //consider
	{
		return fabs(a) <= tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	IRR_FORCE_INLINE bool iszero(const float a, const float tolerance = ROUNDING_ERROR_f32) //consider
	{
		return fabsf(a) <= tolerance;
	}

	//! returns if a equals not zero, taking rounding errors into account
	IRR_FORCE_INLINE bool isnotzero(const float a, const float tolerance = ROUNDING_ERROR_f32) //consider
	{
		return fabsf(a) > tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	IRR_FORCE_INLINE bool iszero(const int32_t a, const int32_t tolerance = 0) //consider
	{
		return ( a & 0x7ffffff ) <= tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	IRR_FORCE_INLINE bool iszero(const uint32_t a, const uint32_t tolerance = 0) //consider
	{
		return a <= tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	IRR_FORCE_INLINE bool iszero(const int64_t a, const int64_t tolerance = 0) //consider
	{
		return abs_(a) <= tolerance;
	}
//depr
	IRR_FORCE_INLINE int32_t s32_min(int32_t a, int32_t b)
	{
		const int32_t mask = (a - b) >> 31;
		return (a & mask) | (b & ~mask);
	}

	IRR_FORCE_INLINE int32_t s32_max(int32_t a, int32_t b)
	{
		const int32_t mask = (a - b) >> 31;
		return (b & mask) | (a & ~mask);
	}

	IRR_FORCE_INLINE int32_t s32_clamp (int32_t value, int32_t low, int32_t high)
	{
		return s32_min(s32_max(value,low), high);
	}

	/*
		float IEEE-754 bit represenation

		0      0x00000000
		1.0    0x3f800000
		0.5    0x3f000000
		3      0x40400000
		+inf   0x7f800000
		-inf   0xff800000
		+NaN   0x7fc00000 or 0x7ff00000
		in general: number = (sign ? -1:1) * 2^(exponent) * 1.(mantissa bits)
	*/

	typedef union { uint32_t u; int32_t s; float f; } inttofloat;
//end depr

    template<typename INT_TYPE>
    inline constexpr bool isNPoT(INT_TYPE value)
    {
        static_assert(std::is_integral<INT_TYPE>::value, "Integral required.");
        return value & (value - static_cast<INT_TYPE>(1));
    }

    template<typename INT_TYPE>
    inline constexpr bool isPoT(INT_TYPE value)
    {
        return !isNPoT<INT_TYPE>(value);
    }

    template<typename INT_TYPE>
    inline int32_t findLSB(INT_TYPE x);
    template<>
    inline int32_t findLSB<uint32_t>(uint32_t x)
    {
#ifdef __GNUC__
        return __builtin_ffs(reinterpret_cast<const int32_t&>(x))-1;
#elif defined(_MSC_VER)
        unsigned long index;
        if (_BitScanForward(&index,reinterpret_cast<const long&>(x)))
            return index;
        return -1;
#endif
    }
    template<>
    inline int32_t findLSB<uint64_t>(uint64_t x)
    {
#ifdef __GNUC__
        return __builtin_ffsl(reinterpret_cast<const int64_t&>(x))-1;
#elif defined(_MSC_VER)
        unsigned long index;
        if (_BitScanForward64(&index,reinterpret_cast<const __int64&>(x)))
            return index;
        return -1;
#endif
    }

    template<typename INT_TYPE>
    inline int32_t findMSB(INT_TYPE x);
    template<>
    inline int32_t findMSB<uint32_t>(uint32_t x)
    {
#ifdef __GNUC__
        return x ? (31-__builtin_clz(x)):(-1);
#elif defined(_MSC_VER)
        unsigned long index;
        if (_BitScanReverse(&index,reinterpret_cast<const long&>(x)))
            return index;
        return -1;
#endif
    }
    template<>
    inline int32_t findMSB<uint64_t>(uint64_t x)
    {
#ifdef __GNUC__
        return x ? (63-__builtin_clzl(x)):(-1);
#elif defined(_MSC_VER)
        unsigned long index;
        if (_BitScanReverse64(&index,reinterpret_cast<const __int64&>(x)))
            return index;
        return -1;
#endif
    }

    template<typename INT_TYPE>
    inline constexpr INT_TYPE roundUpToPoT(INT_TYPE value)
    {
         return INT_TYPE(0x1u)<<INT_TYPE(1+core::findMSB(value-INT_TYPE(1)));
    }

    template<typename INT_TYPE>
    inline constexpr INT_TYPE roundDownToPoT(INT_TYPE value)
    {
        return INT_TYPE(0x1u)<<core::findLSB(value);
    }

    template<typename INT_TYPE>
    inline constexpr INT_TYPE roundUp(INT_TYPE value, INT_TYPE multiple)
    {
        INT_TYPE tmp = (value+multiple-1u)/multiple;
        return tmp*multiple;
    }

    template<typename INT_TYPE>
    inline constexpr INT_TYPE align(INT_TYPE alignment, INT_TYPE size, INT_TYPE& address, INT_TYPE& space)
    {
        INT_TYPE nextAlignedAddr = roundUp(address,alignment);

        INT_TYPE spaceDecrement = nextAlignedAddr-address;
        if (spaceDecrement>space)
            return 0u;

        INT_TYPE newSpace = space-spaceDecrement;
        if (size>newSpace)
            return 0u;

        space = newSpace;
        return address = nextAlignedAddr;
    }


	//! KILL EVERYTHING BELOW???


	#define F32_AS_S32(f)		(*((int32_t *) &(f)))
	#define F32_AS_U32(f)		(*((uint32_t *) &(f)))
	#define F32_AS_U32_POINTER(f)	( ((uint32_t *) &(f)))

	#define F32_VALUE_0		0x00000000
	#define F32_VALUE_1		0x3f800000
	#define F32_SIGN_BIT		0x80000000U
	#define F32_EXPON_MANTISSA	0x7FFFFFFFU

	//! code is taken from IceFPU
	//! Integer representation of a floating-point value.
#ifdef __IRR_FAST_MATH
	#define IR(x)                           ((uint32_t&)(x))
#else
	inline uint32_t IR(float x) {inttofloat tmp; tmp.f=x; return tmp.u;}
#endif

	//! Absolute integer representation of a floating-point value
	#define AIR(x)				(IR(x)&0x7fffffff)

	//! Floating-point representation of an integer value.
#ifdef __IRR_FAST_MATH
	#define FR(x)                           ((float&)(x))
#else
	inline float FR(uint32_t x) {inttofloat tmp; tmp.u=x; return tmp.f;}
	inline float FR(int32_t x) {inttofloat tmp; tmp.s=x; return tmp.f;}
#endif

	//! integer representation of 1.0
	#define IEEE_1_0			0x3f800000
	//! integer representation of 255.0
	#define IEEE_255_0			0x437f0000

#ifdef __IRR_FAST_MATH
	#define	F32_LOWER_0(f)		(F32_AS_U32(f) >  F32_SIGN_BIT)
	#define	F32_LOWER_EQUAL_0(f)	(F32_AS_S32(f) <= F32_VALUE_0)
	#define	F32_GREATER_0(f)	(F32_AS_S32(f) >  F32_VALUE_0)
	#define	F32_GREATER_EQUAL_0(f)	(F32_AS_U32(f) <= F32_SIGN_BIT)
	#define	F32_EQUAL_1(f)		(F32_AS_U32(f) == F32_VALUE_1)
	#define	F32_EQUAL_0(f)		( (F32_AS_U32(f) & F32_EXPON_MANTISSA ) == F32_VALUE_0)

	// only same sign
	#define	F32_A_GREATER_B(a,b)	(F32_AS_S32((a)) > F32_AS_S32((b)))

#else

	#define	F32_LOWER_0(n)		((n) <  0.0f)
	#define	F32_LOWER_EQUAL_0(n)	((n) <= 0.0f)
	#define	F32_GREATER_0(n)	((n) >  0.0f)
	#define	F32_GREATER_EQUAL_0(n)	((n) >= 0.0f)
	#define	F32_EQUAL_1(n)		((n) == 1.0f)
	#define	F32_EQUAL_0(n)		((n) == 0.0f)
	#define	F32_A_GREATER_B(a,b)	((a) > (b))
#endif

#if defined(__BORLANDC__) || defined (__BCPLUSPLUS__)

	// 8-bit bools in borland builder

	//! conditional set based on mask and arithmetic shift
	IRR_FORCE_INLINE uint32_t if_c_a_else_b ( const int8_t condition, const uint32_t a, const uint32_t b )
	{
		return ( ( -condition >> 7 ) & ( a ^ b ) ) ^ b;
	}

	//! conditional set based on mask and arithmetic shift
	IRR_FORCE_INLINE uint32_t if_c_a_else_0 ( const int8_t condition, const uint32_t a )
	{
		return ( -condition >> 31 ) & a;
	}
#else

	//! conditional set based on mask and arithmetic shift
	IRR_FORCE_INLINE uint32_t if_c_a_else_b ( const int32_t condition, const uint32_t a, const uint32_t b )
	{
		return ( ( -condition >> 31 ) & ( a ^ b ) ) ^ b;
	}

	//! conditional set based on mask and arithmetic shift
	IRR_FORCE_INLINE uint16_t if_c_a_else_b ( const int16_t condition, const uint16_t a, const uint16_t b )
	{
		return ( ( -condition >> 15 ) & ( a ^ b ) ) ^ b;
	}

	//! conditional set based on mask and arithmetic shift
	IRR_FORCE_INLINE uint32_t if_c_a_else_0 ( const int32_t condition, const uint32_t a )
	{
		return ( -condition >> 31 ) & a;
	}
#endif

	/*
		if (condition) state |= m; else state &= ~m;
	*/
	IRR_FORCE_INLINE void setbit_cond ( uint32_t &state, int32_t condition, uint32_t mask )
	{
		// 0, or any postive to mask
		//int32_t conmask = -condition >> 31;
		state ^= ( ( -condition >> 31 ) ^ state ) & mask;
	}

	inline float round_( float x )
	{
		return floorf( x + 0.5f );
	}

	// calculate: sqrt ( x )
	IRR_FORCE_INLINE float squareroot(const float f)
	{
		return sqrtf(f);
	}

	// calculate: sqrt ( x )
	IRR_FORCE_INLINE double squareroot(const double f)
	{
		return sqrt(f);
	}

	// calculate: sqrt ( x )
	IRR_FORCE_INLINE int32_t squareroot(const int32_t f)
	{
		return static_cast<int32_t>(squareroot(static_cast<float>(f)));
	}

	// calculate: sqrt ( x )
	IRR_FORCE_INLINE int64_t squareroot(const int64_t f)
	{
		return static_cast<int64_t>(squareroot(static_cast<double>(f)));
	}

	// calculate: 1 / sqrt ( x )
	IRR_FORCE_INLINE double reciprocal_squareroot(const double x)
	{
#if defined ( __IRR_FAST_MATH )
        double result = 1.0 / sqrt(x);
        //! pending perf test
        //_mm_store_sd(&result,_mm_div_sd(_mm_set_pd(0.0,1.0),_mm_sqrt_sd(_mm_load_sd(&x))));
        return result;
#else // no fast math
		return 1.0 / sqrt(x);
#endif
	}

	// calculate: 1 / sqrtf ( x )
	IRR_FORCE_INLINE float reciprocal_squareroot(const float f)
	{
#if defined ( __IRR_FAST_MATH ) && defined ( __IRR_COMPILE_WITH_X86_SIMD_ )
        float result;
        _mm_store_ss(&result,_mm_rsqrt_ps(_mm_load_ss(&f)));
        return result;
#else // no fast math
		return 1.f / sqrtf(f);
#endif
	}

	// calculate: 1 / sqrtf( x )
	IRR_FORCE_INLINE int32_t reciprocal_squareroot(const int32_t x)
	{
		return static_cast<int32_t>(reciprocal_squareroot(static_cast<float>(x)));
	}

	// calculate: 1 / x
	IRR_FORCE_INLINE float reciprocal( const float f )
	{
#if defined (__IRR_FAST_MATH) && defined ( __IRR_COMPILE_WITH_X86_SIMD_ )
        float result;
        _mm_store_ss(&result,_mm_rcp_ps(_mm_load_ss(&f)));
        return result;
#else // no fast math
		return 1.f / f;
#endif
	}

	// calculate: 1 / x
	IRR_FORCE_INLINE double reciprocal ( const double f )
	{
		return 1.0 / f;
	}


	// calculate: 1 / x, low precision allowed
	IRR_FORCE_INLINE float reciprocal_approxim ( const float f )
	{
        //what was here before was not faster
        return reciprocal(f);
    }


	IRR_FORCE_INLINE int32_t floor32(float x)
	{
		return (int32_t) floorf ( x );
	}


	IRR_FORCE_INLINE int32_t ceil32 ( float x )
	{
		return (int32_t) ceilf ( x );
	}



	IRR_FORCE_INLINE int32_t round32(float x)
	{
		return (int32_t) round_(x);
	}

	inline float f32_max3(const float a, const float b, const float c)
	{
		return a > b ? (a > c ? a : c) : (b > c ? b : c);
	}

	inline float f32_min3(const float a, const float b, const float c)
	{
		return a < b ? (a < c ? a : c) : (b < c ? b : c);
	}

	inline float fract ( float x )
	{
		return x - floorf ( x );
	}

} // end namespace core
} // end namespace irr

#ifndef __IRR_FAST_MATH
	using irr::core::IR;
	using irr::core::FR;
#endif

#endif

