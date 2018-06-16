// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MATH_H_INCLUDED__
#define __IRR_MATH_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irrTypes.h"
#include "irrMacros.h"
#include <math.h>
#include <float.h>
#include <stdlib.h> // for abs() etc.
#include <limits.h> // For INT_MAX / UINT_MAX

#if defined(_IRR_SOLARIS_PLATFORM_) || defined(__BORLANDC__) || defined (__BCPLUSPLUS__) || defined (_WIN32_WCE)
	#define sqrtf(X) (float)sqrt((double)(X))
	#define sinf(X) (float)sin((double)(X))
	#define cosf(X) (float)cos((double)(X))
	#define asinf(X) (float)asin((double)(X))
	#define acosf(X) (float)acos((double)(X))
	#define atan2f(X,Y) (float)atan2((double)(X),(double)(Y))
	#define ceilf(X) (float)ceil((double)(X))
	#define floorf(X) (float)floor((double)(X))
	#define powf(X,Y) (float)pow((double)(X),(double)(Y))
	#define fmodf(X,Y) (float)fmod((double)(X),(double)(Y))
	#define fabsf(X) (float)fabs((double)(X))
	#define logf(X) (float)log((double)(X))
#endif

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
	inline float radToDeg(float radians)
	{
		return RADTODEG * radians;
	}

	//! Utility function to convert a radian value to degrees
	/** Provided as it can be clearer to write radToDeg(X) than RADTODEG * X
	\param radians	The radians value to convert to degrees.
	*/
	inline double radToDeg(double radians)
	{
		return RADTODEG64 * radians;
	}

	//! Utility function to convert a degrees value to radians
	/** Provided as it can be clearer to write degToRad(X) than DEGTORAD * X
	\param degrees	The degrees value to convert to radians.
	*/
	inline float degToRad(float degrees)
	{
		return DEGTORAD * degrees;
	}

	//! Utility function to convert a degrees value to radians
	/** Provided as it can be clearer to write degToRad(X) than DEGTORAD * X
	\param degrees	The degrees value to convert to radians.
	*/
	inline double degToRad(double degrees)
	{
		return DEGTORAD64 * degrees;
	}

	//! returns minimum of two values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	inline T min_(const T& a, const T& b)
	{
		return a < b ? a : b;
	}

	//! returns minimum of three values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	inline T min_(const T& a, const T& b, const T& c)
	{
		return a < b ? min_(a, c) : min_(b, c);
	}

	//! returns maximum of two values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	inline T max_(const T& a, const T& b)
	{
		return a < b ? b : a;
	}

	//! returns maximum of three values. Own implementation to get rid of the STL (VS6 problems)
	template<class T>
	inline T max_(const T& a, const T& b, const T& c)
	{
		return a < b ? max_(b, c) : max_(a, c);
	}

	//! returns abs of two values. Own implementation to get rid of STL (VS6 problems)
	template<class T>
	inline T abs_(const T& a)
	{
		return a < (T)0 ? -a : a;
	}

	//! returns linear interpolation of a and b with ratio t
	//! \return: a if t==0, b if t==1, and the linear interpolation else
	template<class T>
	inline T mix(const T& a, const T& b, const float t)
	{
		return a + (b-a)*t;
	}

	//! returns linear interpolation of a and b with ratio t
	//! \return: a if t==0, b if t==1, and the linear interpolation else
	template<class T>
	inline T lerp(const T& a, const T& b, const float t)
	{
		return mix(a,b,t);
	}

	//! clamps a value between low and high
	template <class T>
	inline const T clamp (const T& value, const T& low, const T& high)
	{
		return min_ (max_(value,low), high);
	}

	//! swaps the content of the passed parameters
	// Note: We use the same trick as boost and use two template arguments to
	// avoid ambiguity when swapping objects of an Irrlicht type that has not
	// it's own swap overload. Otherwise we get conflicts with some compilers
	// in combination with stl.
	template <class T1, class T2>
	inline void swap(T1& a, T2& b)
	{
		T1 c(a);
		a = b;
		b = c;
	}

	//! returns if a equals b, taking possible rounding errors into account
	inline bool equals(const double a, const double b, const double tolerance = ROUNDING_ERROR_f64)
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals b, taking possible rounding errors into account
	inline bool equals(const float a, const float b, const float tolerance = ROUNDING_ERROR_f32)
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	union FloatIntUnion32
	{
		FloatIntUnion32(float f1 = 0.0f) : f(f1) {}
		// Portable sign-extraction
		bool sign() const { return (i >> 31) != 0; }

		int32_t i;
		float f;
	};

	//! We compare the difference in ULP's (spacing between floating-point numbers, aka ULP=1 means there exists no float between).
	//\result true when numbers have a ULP <= maxUlpDiff AND have the same sign.
	inline bool equalsByUlp(float a, float b, int maxUlpDiff)
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
	inline bool equals(const int32_t a, const int32_t b)
	{
		return (a == b);
	}

	//! returns if a equals b, not using any rounding tolerance
	inline bool equals(const uint32_t a, const uint32_t b)
	{
		return (a == b);
	}
#endif
	//! returns if a equals b, taking an explicit rounding tolerance into account
	inline bool equals(const int32_t a, const int32_t b, const int32_t tolerance = ROUNDING_ERROR_S32)
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals b, taking an explicit rounding tolerance into account
	inline bool equals(const uint32_t a, const uint32_t b, const int32_t tolerance = ROUNDING_ERROR_S32)
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals b, taking an explicit rounding tolerance into account
	inline bool equals(const int64_t a, const int64_t b, const int64_t tolerance = ROUNDING_ERROR_S64)
	{
		return (a + tolerance >= b) && (a - tolerance <= b);
	}

	//! returns if a equals zero, taking rounding errors into account
	inline bool iszero(const double a, const double tolerance = ROUNDING_ERROR_f64)
	{
		return fabs(a) <= tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	inline bool iszero(const float a, const float tolerance = ROUNDING_ERROR_f32)
	{
		return fabsf(a) <= tolerance;
	}

	//! returns if a equals not zero, taking rounding errors into account
	inline bool isnotzero(const float a, const float tolerance = ROUNDING_ERROR_f32)
	{
		return fabsf(a) > tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	inline bool iszero(const int32_t a, const int32_t tolerance = 0)
	{
		return ( a & 0x7ffffff ) <= tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	inline bool iszero(const uint32_t a, const uint32_t tolerance = 0)
	{
		return a <= tolerance;
	}

	//! returns if a equals zero, taking rounding errors into account
	inline bool iszero(const int64_t a, const int64_t tolerance = 0)
	{
		return abs_(a) <= tolerance;
	}

	inline int32_t s32_min(int32_t a, int32_t b)
	{
		const int32_t mask = (a - b) >> 31;
		return (a & mask) | (b & ~mask);
	}

	inline int32_t s32_max(int32_t a, int32_t b)
	{
		const int32_t mask = (a - b) >> 31;
		return (b & mask) | (a & ~mask);
	}

	inline int32_t s32_clamp (int32_t value, int32_t low, int32_t high)
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

	#define F32_AS_S32(f)		(*((int32_t *) &(f)))
	#define F32_AS_U32(f)		(*((uint32_t *) &(f)))
	#define F32_AS_U32_POINTER(f)	( ((uint32_t *) &(f)))

	#define F32_VALUE_0		0x00000000
	#define F32_VALUE_1		0x3f800000
	#define F32_SIGN_BIT		0x80000000U
	#define F32_EXPON_MANTISSA	0x7FFFFFFFU

	//! code is taken from IceFPU
	//! Integer representation of a floating-point value.
#ifdef IRRLICHT_FAST_MATH
	#define IR(x)                           ((uint32_t&)(x))
#else
	inline uint32_t IR(float x) {inttofloat tmp; tmp.f=x; return tmp.u;}
#endif

	//! Absolute integer representation of a floating-point value
	#define AIR(x)				(IR(x)&0x7fffffff)

	//! Floating-point representation of an integer value.
#ifdef IRRLICHT_FAST_MATH
	#define FR(x)                           ((float&)(x))
#else
	inline float FR(uint32_t x) {inttofloat tmp; tmp.u=x; return tmp.f;}
	inline float FR(int32_t x) {inttofloat tmp; tmp.s=x; return tmp.f;}
#endif

	//! integer representation of 1.0
	#define IEEE_1_0			0x3f800000
	//! integer representation of 255.0
	#define IEEE_255_0			0x437f0000

#ifdef IRRLICHT_FAST_MATH
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


#ifndef REALINLINE
	#ifdef _MSC_VER
		#define REALINLINE __forceinline
	#else
		#define REALINLINE inline
	#endif
#endif

#if defined(__BORLANDC__) || defined (__BCPLUSPLUS__)

	// 8-bit bools in borland builder

	//! conditional set based on mask and arithmetic shift
	REALINLINE uint32_t if_c_a_else_b ( const int8_t condition, const uint32_t a, const uint32_t b )
	{
		return ( ( -condition >> 7 ) & ( a ^ b ) ) ^ b;
	}

	//! conditional set based on mask and arithmetic shift
	REALINLINE uint32_t if_c_a_else_0 ( const int8_t condition, const uint32_t a )
	{
		return ( -condition >> 31 ) & a;
	}
#else

	//! conditional set based on mask and arithmetic shift
	REALINLINE uint32_t if_c_a_else_b ( const int32_t condition, const uint32_t a, const uint32_t b )
	{
		return ( ( -condition >> 31 ) & ( a ^ b ) ) ^ b;
	}

	//! conditional set based on mask and arithmetic shift
	REALINLINE uint16_t if_c_a_else_b ( const int16_t condition, const uint16_t a, const uint16_t b )
	{
		return ( ( -condition >> 15 ) & ( a ^ b ) ) ^ b;
	}

	//! conditional set based on mask and arithmetic shift
	REALINLINE uint32_t if_c_a_else_0 ( const int32_t condition, const uint32_t a )
	{
		return ( -condition >> 31 ) & a;
	}
#endif

	/*
		if (condition) state |= m; else state &= ~m;
	*/
	REALINLINE void setbit_cond ( uint32_t &state, int32_t condition, uint32_t mask )
	{
		// 0, or any postive to mask
		//int32_t conmask = -condition >> 31;
		state ^= ( ( -condition >> 31 ) ^ state ) & mask;
	}

	inline float round_( float x )
	{
		return floorf( x + 0.5f );
	}

	REALINLINE void clearFPUException ()
	{
	}

	// calculate: sqrt ( x )
	REALINLINE float squareroot(const float f)
	{
		return sqrtf(f);
	}

	// calculate: sqrt ( x )
	REALINLINE double squareroot(const double f)
	{
		return sqrt(f);
	}

	// calculate: sqrt ( x )
	REALINLINE int32_t squareroot(const int32_t f)
	{
		return static_cast<int32_t>(squareroot(static_cast<float>(f)));
	}

	// calculate: sqrt ( x )
	REALINLINE int64_t squareroot(const int64_t f)
	{
		return static_cast<int64_t>(squareroot(static_cast<double>(f)));
	}

	// calculate: 1 / sqrt ( x )
	REALINLINE double reciprocal_squareroot(const double x)
	{
#if defined ( IRRLICHT_FAST_MATH )
        double result = 1.0 / sqrt(x);
        //! pending perf test
        //_mm_store_sd(&result,_mm_div_sd(_mm_set_pd(0.0,1.0),_mm_sqrt_sd(_mm_load_sd(&x))));
        return result;
#else // no fast math
		return 1.0 / sqrt(x);
#endif
	}

	// calculate: 1 / sqrtf ( x )
	REALINLINE float reciprocal_squareroot(const float f)
	{
#if defined ( IRRLICHT_FAST_MATH ) && defined ( __IRR_COMPILE_WITH_SSE2 )
        float result;
        _mm_store_ss(&result,_mm_rsqrt_ps(_mm_load_ss(&f)));
        return result;
#else // no fast math
		return 1.f / sqrtf(f);
#endif
	}

	// calculate: 1 / sqrtf( x )
	REALINLINE int32_t reciprocal_squareroot(const int32_t x)
	{
		return static_cast<int32_t>(reciprocal_squareroot(static_cast<float>(x)));
	}

	// calculate: 1 / x
	REALINLINE float reciprocal( const float f )
	{
#if defined (IRRLICHT_FAST_MATH) && defined ( __IRR_COMPILE_WITH_SSE2 )
        float result;
        _mm_store_ss(&result,_mm_rcp_ps(_mm_load_ss(&f)));
        return result;
#else // no fast math
		return 1.f / f;
#endif
	}

	// calculate: 1 / x
	REALINLINE double reciprocal ( const double f )
	{
		return 1.0 / f;
	}


	// calculate: 1 / x, low precision allowed
	REALINLINE float reciprocal_approxim ( const float f )
	{
        //what was here before was not faster
        return reciprocal(f);
    }


	REALINLINE int32_t floor32(float x)
	{
		return (int32_t) floorf ( x );
	}


	REALINLINE int32_t ceil32 ( float x )
	{
		return (int32_t) ceilf ( x );
	}



	REALINLINE int32_t round32(float x)
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

#ifndef IRRLICHT_FAST_MATH
	using irr::core::IR;
	using irr::core::FR;
#endif

#endif

