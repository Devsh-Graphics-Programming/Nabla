// Copyright (C) 2014 Mateusz 'DevSH' Kielan
// This file is part of the "Irrlicht Engine".
// Contributed from "Build a World"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_VECTOR_SIMD_H_INCLUDED__
#define __IRR_VECTOR_SIMD_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef __IRR_COMPILE_WITH_X86_SIMD_

#ifndef __IRR_COMPILE_WITH_SSE2
#error "Either give up on SIMD vectors, check your compiler settings for the -m*sse* flag, or upgrade your CPU"
#endif // __IRR_COMPILE_WITH_SSE2

#include "irrMath.h"
#include "vector2d.h"
#include "vector3d.h"
#include <stdint.h>
#include "SColor.h"

namespace irr
{

namespace video
{
    class SColor;
}

namespace core
{
	class vectorSIMDf;
	template <class T>
	class vectorSIMD_32;
	template <class T>
	class vectorSIMD_16;


    //a class for bitwise shizz
	template <int components> class vectorSIMDBool
    {
    public:
        inline vectorSIMDBool() {_mm_store_ps((float*)value,_mm_setzero_ps());}
		//! These constructors will bytewise cast the reg into the value
		inline vectorSIMDBool(const __m128 &reg) {_mm_store_ps((float*)value,reg);}
		inline vectorSIMDBool(const __m128d &reg) {_mm_store_pd((double*)value,reg);}
		inline vectorSIMDBool(const __m128i &reg) {_mm_store_si128((__m128i*)value,reg);}
		inline vectorSIMDBool(const vectorSIMDBool& other) {_mm_store_ps((float*)value,_mm_load_ps((float*)other.value));}
		//! reads 16 bytes from an array of uint8_t
		inline vectorSIMDBool(uint8_t* const &array) {_mm_store_ps((float*)value,_mm_loadu_ps((float*)array));}
		//! same as above, BUT WILL CRASH IF ARRAY NOT 16 BYTE ALIGNED
		inline vectorSIMDBool(uint8_t* const &array, bool ALIGNED) {_mm_store_ps((float*)value,_mm_load_ps((float*)array));}
		//! Constructor with the same value for all elements
		explicit vectorSIMDBool(const bool &n) {_mm_store_si128((__m128i*)value,n ? _mm_set_epi64x(-0x1ll,-0x1ll):_mm_setzero_si128());}


		inline vectorSIMDBool operator~() const { return _mm_xor_si128(getAsRegister(),_mm_set_epi64x(-0x1ll,-0x1ll)); }
		inline vectorSIMDBool operator&(const vectorSIMDBool &other) const { return _mm_and_si128(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMDBool operator|(const vectorSIMDBool &other) const { return _mm_or_si128(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMDBool operator^(const vectorSIMDBool &other) const { return _mm_xor_si128(getAsRegister(),other.getAsRegister()); }
/*
NO BITSHIFTING SUPPORT
*/
		inline vectorSIMDBool<components> operator!() const { return vectorSIMDBool<components>(); }
		inline vectorSIMDBool<components> operator&&(const vectorSIMDBool<components> &other) const { return vectorSIMDBool<components>(); }
		inline vectorSIMDBool<components> operator||(const vectorSIMDBool<components> &other) const { return vectorSIMDBool<components>(); }


		//! like GLSL, returns true if any bit of value is set
		inline bool any(void) const
		{
		    return ((uint64_t*)value)[0]|((uint64_t*)value)[1];
		}

        //! like GLSL, returns true if all bits of value are set
        inline bool allBits(void) const
        {
            return (((uint64_t*)value)[0]&((uint64_t*)value)[1])==0xffffffffffffffffull;
        }
        //! like GLSL, returns true if all components non zero
        inline bool all(void) const
        {
            return 0;
        }


        //! in case you want to do your own SSE
        inline __m128i getAsRegister() const {return _mm_load_si128((__m128i*)value);}


#ifdef _IRR_WINDOWS_
        __declspec(align(SIMD_ALIGNMENT)) uint8_t value[16];
    };
#else
	uint8_t value[16];
    } __attribute__ ((__aligned__(SIMD_ALIGNMENT)));
#endif

	//! partial specialization for variable width vectors
	template <>
	inline bool vectorSIMDBool<2>::all(void) const
    {
        return (((uint64_t*)value)[0]&&((uint64_t*)value)[1]);
    }
    template <>
	inline bool vectorSIMDBool<4>::all(void) const
    {
        return ((uint32_t*)value)[0]&&((uint32_t*)value)[1]&&((uint32_t*)value)[2]&&((uint32_t*)value)[3];
    }
    template <>
	inline bool vectorSIMDBool<8>::all(void) const
    {
        __m128i xmm0 = _mm_xor_si128(_mm_cmpeq_epi16(getAsRegister(),_mm_setzero_si128()),_mm_set_epi16(-1,-1,-1,-1,-1,-1,-1,-1));
        xmm0 = _mm_and_si128(xmm0,_mm_shuffle_epi32(xmm0,_MM_SHUFFLE(0,1,2,3))); // (0&&6,1&&7,  2&&4,3&&5,  ...)
        xmm0 = _mm_and_si128(xmm0,_mm_shufflelo_epi16(xmm0,_MM_SHUFFLE(1,0,3,2))); // (0&&2&&4&&6, 1&&3&&5&&7, ... )
        uint16_t tmpStorage[2];
        _mm_store_ss((float*)tmpStorage,_mm_castsi128_ps(xmm0));
        return tmpStorage[0]&tmpStorage[1];
    }/*
    template <>
	inline bool vectorSIMDBool<16>::all(void) const
    {
        __m128i xmm0 = _mm_xor_si128(_mm_cmpeq_epi8(getAsRegister(),_mm_setzero_si128()),_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
        xmm0 = _mm_and_si128(xmm0,_mm_shuffle_epi32(xmm0,_MM_SHUFFLE(0,1,2,3))); // (0&&12,1&&13,2&&14,3&&15,  4&&8,5&&9,6&&10,7&&11,  ...)
        xmm0 = _mm_and_si128(xmm0,_mm_shufflelo_epi16(xmm0,_MM_SHUFFLE(1,0,3,2))); // (0&&4&&8&&12,1&&5&&9&&13,2&&6&&10&&14,3&&7&&11&&15,  ...)
        xmm0 = _mm_and_si128(xmm0,_mm_slli_si128(xmm0,2)); // (even &&,odd &&,  ...)
        _mm_store_si128((__m128i*)tmpStorage,xmm0);
        return tmpStorage[0]&&tmpStorage[1];
    }

    //! following do ANDs (not bitwise ANDs)
    /*
    template <>
	inline vectorSIMDBool<2> vectorSIMDBool<2>::operator&&(const vectorSIMDBool<2> &other) const
    {
    }
    template <>
	inline vectorSIMDBool<2> vectorSIMDBool<2>::operator||(const vectorSIMDBool<2> &other) const
    {
    }
    template <>
	inline vectorSIMDBool<2> vectorSIMDBool<2>::operator!() const
    {
    }*/
    template <>
	inline vectorSIMDBool<4> vectorSIMDBool<4>::operator&&(const vectorSIMDBool<4> &other) const
    {
        __m128 xmm0 = _mm_and_ps(_mm_cmpneq_ps(_mm_castsi128_ps(other.getAsRegister()),_mm_setzero_ps()),_mm_cmpneq_ps(_mm_castsi128_ps(getAsRegister()),_mm_setzero_ps()));
        return vectorSIMDBool<4>(xmm0);
    }
    template <>
	inline vectorSIMDBool<4> vectorSIMDBool<4>::operator||(const vectorSIMDBool<4> &other) const
    {
        __m128i xmm0 = _mm_or_si128(other.getAsRegister(),getAsRegister());
        return vectorSIMDBool<4>(_mm_cmpneq_ps(_mm_castsi128_ps(xmm0),_mm_setzero_ps()));
    }
    template <>
	inline vectorSIMDBool<4> vectorSIMDBool<4>::operator!() const
    {
        return vectorSIMDBool<4>(_mm_cmpeq_ps(_mm_castsi128_ps(getAsRegister()),_mm_setzero_ps()));
    }
    template <>
	inline vectorSIMDBool<8> vectorSIMDBool<8>::operator&&(const vectorSIMDBool<8> &other) const
    {
        __m128i xmm0 = _mm_andnot_si128(_mm_cmpeq_epi16(other.getAsRegister(),_mm_setzero_si128()),_mm_xor_si128(_mm_cmpeq_epi16(getAsRegister(),_mm_setzero_si128()),_mm_set_epi16(-1,-1,-1,-1,-1,-1,-1,-1)));
        return vectorSIMDBool<8>(xmm0);
    }
    template <>
	inline vectorSIMDBool<8> vectorSIMDBool<8>::operator||(const vectorSIMDBool<8> &other) const
    {
        __m128i xmm0 = _mm_or_si128(other.getAsRegister(),getAsRegister());
        return vectorSIMDBool<8>(_mm_xor_si128(_mm_cmpeq_epi16(xmm0,_mm_setzero_si128()),_mm_set_epi16(-1,-1,-1,-1,-1,-1,-1,-1)));
    }
    template <>
	inline vectorSIMDBool<8> vectorSIMDBool<8>::operator!() const
    {
        return vectorSIMDBool<8>(_mm_cmpeq_epi16(getAsRegister(),_mm_setzero_si128()));
    }/*
    template <>
	inline vectorSIMDBool<16> vectorSIMDBool<16>::operator&&(const vectorSIMDBool<16> &other) const
    {
    }
    template <>
	inline vectorSIMDBool<16> vectorSIMDBool<16>::operator||(const vectorSIMDBool<16> &other) const
    {
    }
    template <>
	inline vectorSIMDBool<16> vectorSIMDBool<16>::operator!() const
    {
    }*/


    //! Typedef for N-bit wide boolean vectors
	//typedef vectorSIMDBool<16> vector16db_SIMD;
	typedef vectorSIMDBool<8> vector8db_SIMD;
	typedef vectorSIMDBool<4> vector4db_SIMD;
	//typedef vectorSIMDBool<2> vector2db_SIMD;


#include "SIMDswizzle.h"


    inline vectorSIMDf abs(const vectorSIMDf& a);
    inline vectorSIMDf ceil(const vectorSIMDf& a);
    inline vectorSIMDf clamp(const vectorSIMDf& value, const vectorSIMDf& low, const vectorSIMDf& high);
	inline vectorSIMDf cross(const vectorSIMDf& a, const vectorSIMDf& b);
    inline vectorSIMDf degToRad(const vectorSIMDf& degrees);
	inline vectorSIMDf dot(const vectorSIMDf& a, const vectorSIMDf& b);
    inline vector4db_SIMD equals(const vectorSIMDf& a,const vectorSIMDf& b, const float tolerance = ROUNDING_ERROR_f32);
    inline vectorSIMDf floor(const vectorSIMDf& a);
    inline vectorSIMDf fract(const vectorSIMDf& a);
    inline vectorSIMDf inversesqrt(const vectorSIMDf& a);
    inline vectorSIMDf length(const vectorSIMDf& v);
    inline vectorSIMDf lerp(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& t);
    inline vectorSIMDf mix(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& t);
    inline vectorSIMDf normalize(const vectorSIMDf& v);
    inline vectorSIMDf radToDeg(const vectorSIMDf& radians);
    inline vectorSIMDf reciprocal(const vectorSIMDf& a);
    inline vectorSIMDf sqrt(const vectorSIMDf& a);


#ifdef _IRR_WINDOWS_
    __declspec(align(SIMD_ALIGNMENT)) class vectorSIMDf : public SIMD_32bitSwizzleAble<vectorSIMDf,__m128>
#else
    class vectorSIMDf : public SIMD_32bitSwizzleAble<vectorSIMDf,__m128>
#endif
	{
	public:
		//! Default constructor (null vector).
		inline vectorSIMDf() {_mm_store_ps(pointer,_mm_setzero_ps());}
        //! Constructor with four different values, FASTEST IF the values are constant literals
		//yes this is correct usage with _mm_set_**(), due to little endianness the thing gets set in "reverse" order
		inline explicit vectorSIMDf(const float &nx, const float &ny, const float &nz, const float &nw) {_mm_store_ps(pointer,_mm_set_ps(nw,nz,ny,nx));}
		//! 3d constructor
		inline explicit vectorSIMDf(const float &nx, const float &ny, const float &nz) {_mm_store_ps(pointer,_mm_set_ps(0.f,nz,ny,nx));}
		//! 2d constructor
		inline explicit vectorSIMDf(const float &nx, const float &ny) {_mm_store_ps(pointer,_mm_set_ps(0.f,0.f,ny,nx));}
		//! Fast Constructor from floats, they come in normal order [0]=X,[1]=Y, etc.
		inline vectorSIMDf(const float* const &array) {_mm_store_ps(pointer,_mm_loadu_ps(array));}
		//! Fastest Constructor from floats, they come in normal order [0]=X,[1]=Y, etc.
		//! Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMDf(const float* const &array, bool ALIGNED) {_mm_store_ps(pointer,_mm_load_ps(array));}
		//! Fastest and most natural constructor
		inline vectorSIMDf(const __m128 &reg) {_mm_store_ps(pointer,reg);}
		//! Constructor with the same value for all elements
		inline explicit vectorSIMDf(const float &n) {_mm_store_ps(pointer,_mm_load_ps1(&n));}
		//! Copy constructor
		inline vectorSIMDf(const vectorSIMDf& other) {_mm_store_ps(pointer,other.getAsRegister());}

/**
        static inline void* operator new(size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void operator delete(void* ptr)
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new[](size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void  operator delete[](void* ptr) throw()
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new(std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete(void* p,void* t) throw() {}
        static inline void* operator new[](std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete[](void* p,void* t) throw() {}
**/

/*
		inline vectorSIMDf(const vectorSIMDu32& other);
		inline vectorSIMDf(const vectorSIMDi32& other);
		inline vectorSIMDf(const vectorSIMDu16& other);
		inline vectorSIMDf(const vectorSIMDi16& other);
**/

		inline vectorSIMDf& operator=(const vectorSIMDf& other) { _mm_store_ps(pointer,other.getAsRegister()); return *this; }

        //! bitwise ops
        inline vectorSIMDf operator&(const vectorSIMDf& other) {return _mm_and_ps(getAsRegister(),other.getAsRegister());}
        inline vectorSIMDf operator|(const vectorSIMDf& other) {return _mm_or_ps(getAsRegister(),other.getAsRegister());}
        inline vectorSIMDf operator^(const vectorSIMDf& other) {return _mm_xor_ps(getAsRegister(),other.getAsRegister());}

        //! in case you want to do your own SSE
        inline __m128 getAsRegister() const {return _mm_load_ps(pointer);}


		// operators against vectors
		inline vectorSIMDf operator-() const { return _mm_xor_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80000000u)),getAsRegister()); }

		inline vectorSIMDf operator+(const vectorSIMDf& other) const { return _mm_add_ps(other.getAsRegister(),getAsRegister()); }
		inline vectorSIMDf& operator+=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_add_ps(other.getAsRegister(),getAsRegister())); return *this; }

		inline vectorSIMDf operator-(const vectorSIMDf& other) const { return _mm_sub_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMDf& operator-=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_sub_ps(getAsRegister(),other.getAsRegister())); return *this; }

		inline vectorSIMDf operator*(const vectorSIMDf& other) const { return _mm_mul_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMDf& operator*=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_mul_ps(getAsRegister(),other.getAsRegister())); return *this; }
#ifdef IRRLICHT_FAST_MATH
		inline vectorSIMDf operator/(const vectorSIMDf& other) const { return _mm_mul_ps(getAsRegister(),_mm_rcp_ps(other.getAsRegister())); }
		inline vectorSIMDf& operator/=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_mul_ps(getAsRegister(),_mm_rcp_ps(other.getAsRegister()))); return *this; }
#else
		inline vectorSIMDf operator/(const vectorSIMDf& other) const { return preciseDivision(other); }
		inline vectorSIMDf& operator/=(const vectorSIMDf& other) { (*this) = preciseDivision(other); return *this; }
#endif
		inline vectorSIMDf preciseDivision(const vectorSIMDf& other) const { return _mm_div_ps(getAsRegister(),other.getAsRegister()); }


		//operators against scalars
		inline vectorSIMDf  operator+(const float &val) const { return (*this)+vectorSIMDf(val); }
		inline vectorSIMDf& operator+=(const float &val) { return ( (*this) += vectorSIMDf(val) ); }

		inline vectorSIMDf operator-(const float &val) const { return (*this)-vectorSIMDf(val); }
		inline vectorSIMDf& operator-=(const float &val) { return ( (*this) -= vectorSIMDf(val) ); }

		inline vectorSIMDf  operator*(const float &val) const { return (*this)*vectorSIMDf(val); }
		inline vectorSIMDf& operator*=(const float &val) { return ( (*this) *= vectorSIMDf(val) ); }

#ifdef IRRLICHT_FAST_MATH
		inline vectorSIMDf operator/(const float &v) const { return vectorSIMDf(_mm_mul_ps(_mm_rcp_ps(_mm_load_ps1(&v)),getAsRegister())); }
		inline vectorSIMDf& operator/=(const float &v) { _mm_store_ps(pointer,_mm_mul_ps(_mm_rcp_ps(_mm_load_ps1(&v)),getAsRegister())); return *this; }
#else
		inline vectorSIMDf operator/(const float &v) const { return vectorSIMDf(_mm_div_ps(getAsRegister(),_mm_load_ps1(&v))); }
		inline vectorSIMDf& operator/=(const float &v) { _mm_store_ps(pointer,_mm_div_ps(getAsRegister(),_mm_load_ps1(&v))); return *this; }
#endif

		//! I AM BREAKING IRRLICHT'S COMPARISON OPERATORS
		inline vector4db_SIMD operator<=(const vectorSIMDf& other) const
		{
		    return _mm_cmple_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator>=(const vectorSIMDf& other) const
		{
		    return _mm_cmpge_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator<(const vectorSIMDf& other) const
		{
		    return _mm_cmplt_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator>(const vectorSIMDf& other) const
		{
		    return _mm_cmpgt_ps(getAsRegister(),other.getAsRegister());
		}

		//! only the method that returns bool confirms if two vectors are exactly the same
		inline vector4db_SIMD operator==(const vectorSIMDf& other) const
		{
			return _mm_cmpeq_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator!=(const vectorSIMDf& other) const
		{
			return _mm_cmpneq_ps(getAsRegister(),other.getAsRegister());
		}



		// functions
		//! zeroes out out of range components (useful before performing a dot product so it doesnt get polluted with random values)
		//! WARNING IT DOES COST CYCLES
		inline void makeSafe2D(void) {_mm_store_ps(pointer,_mm_and_ps(_mm_load_ps(pointer),_mm_castsi128_ps(_mm_set_epi32(0,0,-1,-1))));}
		inline void makeSafe3D(void) {_mm_store_ps(pointer,_mm_and_ps(_mm_load_ps(pointer),_mm_castsi128_ps(_mm_set_epi32(0,-1,-1,-1))));}

		//! slightly faster than memcpy'ing into the pointers
		inline vectorSIMDf& set(float* const &array) {_mm_store_ps(pointer,_mm_loadu_ps(array)); return *this;}
		//! FASTEST WAY TO SET VALUES, Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMDf& set(float* const &array, bool ALIGNED) {_mm_store_ps(pointer,_mm_load_ps(array)); return *this;}
		//! normal set() like vector3df's, but for different dimensional vectors
		inline vectorSIMDf& set(const float &nx, const float &ny, const float &nz, const float &nw) {_mm_store_ps(pointer,_mm_set_ps(nw,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(const float &nx, const float &ny, const float &nz) {_mm_store_ps(pointer,_mm_set_ps(0.f,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(const float &nx, const float &ny) {_mm_store_ps(pointer,_mm_set_ps(0.f,0.f,ny,nx)); return *this;}
		inline vectorSIMDf& set(const vectorSIMDf& p) {_mm_store_ps(pointer,p.getAsRegister()); return *this;}
		//! convert from vectorNdf types of irrlicht - it will read a few values past the range of the allocated memory but _mm_loadu_ps shouldnt have that kind of protection
		inline vectorSIMDf& set(const vector3df &p) {_mm_store_ps(pointer,_mm_loadu_ps(&p.X)); makeSafe3D(); return *this;}
		inline vectorSIMDf& set(const vector2df &p) {_mm_store_ps(pointer,_mm_loadu_ps(&p.X)); makeSafe2D(); return *this;}

        //! going directly from vectorSIMD to irrlicht types is safe cause vectorSIMDf is wider
		inline vector2df getAsVector2df(void) const
		{
		    return *((vector2df*)pointer);
		}
		inline vector3df getAsVector3df(void) const
		{
		    return *((vector3df*)pointer);
		}

		inline void storeTo4Floats(float* out) const
		{
		    _mm_storeu_ps(out,_mm_load_ps(pointer));
		}
		inline void storeTo4Floats(float* out, bool ALIGNED) const
		{
		    _mm_store_ps(out,_mm_load_ps(pointer));
		}


		//! Get length of the vector.
		inline float getLengthAsFloat() const
		{
		    float result;
		    _mm_store_ss(&result,length(*this).getAsRegister());
		    return result;
        }
        //! Useful when you have to divide a vector by another vector's length (so you dont convert/store to a scalar)
        //! all components are filled with length
        //! if you need something else, you can get the register and shuffle
		inline vectorSIMDf getLength() const
		{
		    return length(*this);
		}


        inline vectorSIMDf getSquareRoot() const
        {
            return _mm_sqrt_ps(getAsRegister());
        }

        inline vectorSIMDf getReciprocalSQRT() const
        {
            return _mm_rsqrt_ps(getAsRegister());
        }

		//! Get the dot product with another vector.
		inline float dotProductAsFloat(const vectorSIMDf& other) const
		{
		    float result;
		    _mm_store_ss(&result,dot(*this,other).getAsRegister());
		    return result;
		}
		inline vectorSIMDf dotProduct(const vectorSIMDf& other) const
		{
            return dot(*this,other);
		}

		//! Get squared length of the vector.
		/** This is useful because it is much faster than getLength().
		\return Squared length of the vector. */
		inline float getLengthSQAsFloat() const
		{
		    float result;
		    _mm_store_ss(&result,dotProduct(*this).getAsRegister());
		    return result;
		}
        //! Useful when you have to divide a vector by another vector's length (so you dont convert/store to a scalar)
		inline vectorSIMDf getLengthSQ() const
		{
		    return dot(*this,*this);
        }


		//! Get distance from another point.
		/** Here, the vector is interpreted as point in 3 dimensional space. */
		inline float getDistanceFromAsFloat(const vectorSIMDf& other) const
		{
		    float result;
		    _mm_store_ss(&result,((*this)-other).getLength().getAsRegister());
		    return result;
		}
        inline vectorSIMDf getDistanceFrom(const vectorSIMDf& other) const
		{
			return ((*this)-other).getLength();
		}

		//! Returns squared distance from another point.
		/** Here, the vector is interpreted as point in 3 dimensional space. */
		inline float getDistanceFromSQAsFloat(const vectorSIMDf& other) const
		{
		    float result;
		    _mm_store_ss(&result,((*this)-other).getLengthSQ().getAsRegister());
		    return result;
		}
        inline vectorSIMDf getDistanceFromSQ(const vectorSIMDf& other) const
		{
			return ((*this)-other).getLengthSQ();
		}

		//! Calculates the cross product with another vector.
		/** \param p Vector to multiply with.
		\return Crossproduct of this vector with p. */
		inline vectorSIMDf crossProduct(const vectorSIMDf& p) const
		{
		    return cross(*this,p);
		}

		//! Sets the length of the vector to a new value
		inline vectorSIMDf& setLengthAsFloat(float newlength)
		{
			(*this) = normalize(*this)*newlength;
			return (*this);
		}

		//! Inverts the vector.
		inline vectorSIMDf& invert()
		{
			_mm_store_ps(pointer,_mm_xor_ps(_mm_castsi128_ps(_mm_set_epi32(0x80000000u,0x80000000u,0x80000000u,0x80000000u)),getAsRegister()));
			return *this;
		}
		//! Returns component-wise absolute value of itself
		inline vectorSIMDf getAbsoluteValue() const
		{
			return abs(*this);
		}

		//! Rotates the vector by a specified number of RADIANS around the Y axis and the specified center.
		/** \param radians Number of RADIANS to rotate around the Y axis.
		\param center The center of the rotation. */
		inline void rotateXZByRAD(const float &radians, const vectorSIMDf& center)
		{
			__m128 xmm1 = center.getAsRegister();
			__m128 xmm0 = _mm_sub_ps(getAsRegister(),xmm1);

            float cs = cosf(radians);
            float sn = sinf(radians);
			__m128 xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0); // now contains (X*cos,radom_crap,Z*cos,random_crap)
			__m128 xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,0,1,2))); // now contains (Z*sin,radom_crap,X*cos,random_crap)
			xmm3 = _mm_xor_ps(xmm3,_mm_castsi128_ps(_mm_set_epi32(0,0,0,0x80000000u))); // invert the Z*sin
			xmm0 = _mm_add_ps(_mm_add_ps(xmm2,xmm3),xmm1); // gives us ((X*cs - Z*sn), (X*cs - Z*sn), (X*sn + Z*cs), (X*sn + Z*cs))

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0),_mm_set_epi32(0,-1,0,-1),(char*)pointer);// only overwrites the X,Z elements of our vector
		}
		inline void rotateXZByRAD(const float &radians)
		{
			__m128 xmm0 = getAsRegister();

            float cs = cosf(radians);
            float sn = sinf(radians);
			__m128 xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0); // now contains (X*cos,radom_crap,Z*cos,random_crap)
			__m128 xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,0,1,2))); // now contains (Z*sin,radom_crap,X*cos,random_crap)
			xmm3 = _mm_xor_ps(xmm3,_mm_castsi128_ps(_mm_set_epi32(0,0,0,0x80000000u))); // invert the Z*sin
			xmm0 = _mm_add_ps(xmm2,xmm3); // gives us ((X*cs - Z*sn), (X*cs - Z*sn), (X*sn + Z*cs), (X*sn + Z*cs))

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0),_mm_set_epi32(0,-1,0,-1),(char*)pointer);// only overwrites the X,Z elements of our vector
		}

		//! Rotates the vector by a specified number of RADIANS around the Z axis and the specified center.
		/** \param RADIANS: Number of RADIANS to rotate around the Z axis.
		\param center: The center of the rotation. */
		inline void rotateXYByRAD(const float &radians, const vectorSIMDf& center)
		{
			__m128 xmm1 = center.getAsRegister();
			__m128 xmm0 = _mm_sub_ps(getAsRegister(),xmm1);

            float cs = cosf(radians);
            float sn = sinf(radians);
			__m128 xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0); // now contains (X*cos,Y*cos,...,...)
			__m128 xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,2,0,1))); // now contains (Y*sin,X*cos,...)
			xmm3 = _mm_xor_ps(xmm3,_mm_castsi128_ps(_mm_set_epi32(0,0,0,0x80000000u))); // invert the Y*sin
			xmm0 = _mm_add_ps(_mm_add_ps(xmm2,xmm3),xmm1); // gives us ((X*cs - Y*sn), (Y*cs + X*sn),...,...)

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0),_mm_set_epi32(0,0,-1,-1),(char*)pointer);// only overwrites the X,Y elements of our vector
		}
		inline void rotateXYByRAD(const float &radians)
		{
			__m128 xmm0 = getAsRegister();

            float cs = cosf(radians);
            float sn = sinf(radians);
			__m128 xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0); // now contains (X*cos,Y*cos,...,...)
			__m128 xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,2,0,1))); // now contains (Y*sin,X*sin,...)
			xmm3 = _mm_xor_ps(xmm3,_mm_castsi128_ps(_mm_set_epi32(0,0,0,0x80000000u))); // invert the Y*sin
			xmm0 = _mm_add_ps(xmm2,xmm3); // gives us ((X*cs - Y*sn), (Y*cs + X*sn),...,...)

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0),_mm_set_epi32(0,0,-1,-1),(char*)pointer);// only overwrites the X,Y elements of our vector
		}

		//! Rotates the vector by a specified number of degrees around the X axis and the specified center.
		/** \param degrees: Number of degrees to rotate around the X axis.
		\param center: The center of the rotation. */
		inline void rotateYZByRAD(const float &radians, const vectorSIMDf& center)
		{
			__m128 xmm1 = center.getAsRegister();
			__m128 xmm0 = _mm_sub_ps(getAsRegister(),xmm1);

            float cs = cosf(radians);
            float sn = sinf(radians);
			__m128 xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0); // now contains (X*cos,Y*cos,...,...)
			__m128 xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,1,2,0))); // now contains (...,Z*sin,Y*sin,...)
			xmm3 = _mm_xor_ps(xmm3,_mm_castsi128_ps(_mm_set_epi32(0,0,0x80000000u,0))); // invert the Z*sin
			xmm0 = _mm_add_ps(_mm_add_ps(xmm2,xmm3),xmm1); // gives us ((X*cs - Y*sn), (Y*cs + X*sn),...,...)

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0),_mm_set_epi32(0,-1,-1,0),(char*)pointer);// only overwrites the X,Y elements of our vector
		}
		inline void rotateYZByRAD(const float &radians)
		{
			__m128 xmm0 = getAsRegister();

            float cs = cosf(radians);
            float sn = sinf(radians);
			__m128 xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0); // now contains (X*cos,Y*cos,...,...)
			__m128 xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,1,2,0))); // now contains (...,Z*sin,Y*sin,...)
			xmm3 = _mm_xor_ps(xmm3,_mm_castsi128_ps(_mm_set_epi32(0,0,0x80000000u,0))); // invert the Z*sin
			xmm0 = _mm_add_ps(xmm2,xmm3); // gives us ((X*cs - Y*sn), (Y*cs + X*sn),...,...)

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0),_mm_set_epi32(0,-1,-1,0),(char*)pointer);// only overwrites the X,Y elements of our vector
		}



		//! Get the rotations that would make a (0,0,1) direction vector point in the same direction as this direction vector.
		/* Thanks to Arras on the Irrlicht forums for this method.  This utility method is very useful for
		orienting scene nodes towards specific targets.  For example, if this vector represents the difference
		between two scene nodes, then applying the result of getHorizontalAngle() to one scene node will point
		it at the other one.
		Example code:
		// Where target and seeker are of type ISceneNode*
		const vector3df toTarget(target->getAbsolutePosition() - seeker->getAbsolutePosition());
		const vector3df requiredRotation = toTarget.getHorizontalAngle();
		seeker->setRotation(requiredRotation);

		\return A rotation vector containing the X (pitch) and Y (raw) rotations (in degrees) that when applied to a
		+Z (e.g. 0, 0, 1) direction vector would make it point in the same direction as this vector. The Z (roll) rotation
		is always 0, since two Euler rotations are sufficient to point in any given direction. *
		inline vectorSIMDf getHorizontalAngle3D() const
		{
			vectorSIMDf angle;

			const float tmp = atan2f(x,z);
			angle.y = tmp;

            __m128 xmm0 = ((*this)*(*this)).getAsRegister();
			xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,0,1,2)));
			float z1;
			_mm_store_ss(&z1,_mm_sqrt_ss(xmm0));

			angle.x = atan2f(z1, y) - core::PI*0.5f;

			return angle;
		}

		//! Get the spherical coordinate angles, can we do 4-sphere coordinates
		/** This returns Euler radians for the point represented by
		this vector.
		*
		inline vectorSIMDf getSphericalCoordinates3D() const
		{
			vectorSIMDf angle = *this;
			angle.makeSafe3D();
			angle = angle.getLength();

			if (angle.w) //doesnt matter which component
			{
				if (X!=0)
				{
					angle.Y = atan2f(Z,X);
				}
				else if (Z<0)
					angle.Y=180;

				angle.X = (T)(acos(Y * core::reciprocal_squareroot(length)) * RADTODEG64);
			}
			else
                return vectorSIMDf(0.f);
		}

		//! Builds a direction vector from (this) rotation vector.
		/** This vector is assumed to be a rotation vector composed of 3 Euler angle rotations, in degrees.
		The implementation performs the same calculations as using a matrix to do the rotation.

		\param[in] forwards  The direction representing "forwards" which will be rotated by this vector.
		If you do not provide a direction, then the +Z axis (0, 0, 1) will be assumed to be forwards.
		\return A direction vector calculated by rotating the forwards direction by the 3 Euler angles
		(in degrees) represented by this vector. *
		inline vectorSIMDf rotationToDirection3D() const
		{
			const float cr = cosf( x );
			const float sr = sinf( x );
			const float cp = cosf( y );
			const float sp = sinf( y );
			const float cy = cosf( z );
			const float sy = sinf( z );

			const float crsp = cr*sp;

			return vectorSIMDf(( crsp*cy+sr*sy ), ( crsp*sy-sr*cy ), ( cr*cp ),0);
		}
		inline vectorSIMDf rotationToDirection3D(const vectorSIMDf &forwards = vectorSIMDf(0, 0, 1, 0)) const
		{
			const float cr = cosf( x );
			const float sr = sinf( x );
			const float cp = cosf( y );
			const float sp = sinf( y );
			const float cy = cosf( z );
			const float sy = sinf( z );

			const float crsp = cr*sp;
			const float srsp = sr*sp;

			const double pseudoMatrix[] = {
				( cp*cy ), ( cp*sy ), ( -sp ),
				( srsp*cy-cr*sy ), ( srsp*sy+cr*cy ), ( sr*cp ),
				( crsp*cy+sr*sy ), ( crsp*sy-sr*cy ), ( cr*cp )};

			return vector3d<T>(
				(T)(forwards.X * pseudoMatrix[0] +
					forwards.Y * pseudoMatrix[3] +
					forwards.Z * pseudoMatrix[6]),
				(T)(forwards.X * pseudoMatrix[1] +
					forwards.Y * pseudoMatrix[4] +
					forwards.Z * pseudoMatrix[7]),
				(T)(forwards.X * pseudoMatrix[2] +
					forwards.Y * pseudoMatrix[5] +
					forwards.Z * pseudoMatrix[8]));
		}*/

        static inline vectorSIMDf fromSColor(const video::SColor &col)
        {
            vectorSIMDf retVal;

            __m128i xmm0 = _mm_castps_si128(_mm_load_ss((float*)&col));
            xmm0 = _mm_unpacklo_epi8(xmm0,_mm_setzero_si128());
            xmm0 = _mm_unpacklo_epi16(xmm0,_mm_setzero_si128());
            __m128 xmm1 = _mm_div_ps(_mm_cvtepi32_ps(xmm0),_mm_set_ps(255.f,255.f,255.f,255.f));
            xmm1 = FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,0,1,2));
            _mm_store_ps(retVal.pointer,xmm1);

            return retVal;
        }


        union
        {
            struct{
                float X; float Y; float Z; float W;
            };
            struct{
                float x; float y; float z; float w;
            };
            struct{
                float r; float g; float b; float a;
            };
            struct{
                float s; float t; float p; float q;
            };
            float pointer[4];
        };
#ifdef _IRR_WINDOWS_
    };
#else
    } __attribute__ ((__aligned__(SIMD_ALIGNMENT)));
#endif


    //! Returns component-wise absolute value of a
    inline vectorSIMDf abs(const vectorSIMDf& a)
    {
        return _mm_and_ps(a.getAsRegister(),_mm_castsi128_ps(_mm_set_epi32(0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF)));
    }
	 inline vectorSIMDf radToDeg(const vectorSIMDf& radians)
	{
	    return radians*vectorSIMDf(RADTODEG);
	}
     inline vectorSIMDf degToRad(const vectorSIMDf& degrees)
    {
        return degrees*vectorSIMDf(DEGTORAD);
    }
     inline vectorSIMDf mix(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& t)
    {
        return a+(b-a)*t;
    }
     inline vectorSIMDf lerp(const vectorSIMDf& a, const vectorSIMDf& b, const vectorSIMDf& t)
    {
        return mix(a,b,t);
	}
	template<>
	inline vectorSIMDf max_(const vectorSIMDf& a, const vectorSIMDf& b)
    {
        return _mm_max_ps(a.getAsRegister(),b.getAsRegister());
    }
	template<>
	inline vectorSIMDf min_(const vectorSIMDf& a, const vectorSIMDf& b)
    {
        return _mm_min_ps(a.getAsRegister(),b.getAsRegister());
    }
    inline vectorSIMDf clamp(const vectorSIMDf& value, const vectorSIMDf& low, const vectorSIMDf& high)
    {
        return min_(max_(value,low),high);
    }
    inline vector4db_SIMD equals(const vectorSIMDf& a,const vectorSIMDf& b, const float tolerance)
    {
        return (a + tolerance >= b) && (a - tolerance <= b);
    }
    inline vectorSIMDf floor(const vectorSIMDf& a)
    {
        vectorSIMDf b = a;
        vector4db_SIMD notTooLarge = b.getAbsoluteValue()<vectorSIMDf(float(0x800000)); //cutoff point for flooring
        __m128i xmm0 = _mm_cvtps_epi32(b.getAsRegister());
        __m128 xmm1 = _mm_cvtepi32_ps(xmm0);

        xmm1 =  _mm_add_ps(xmm1, _mm_and_ps(_mm_cmpgt_ps(xmm1, a.getAsRegister()), _mm_set1_ps(-1.f)));

        _mm_maskmoveu_si128(_mm_castps_si128(xmm1),notTooLarge.getAsRegister(),(char*)b.pointer);
        return b;
    }
    inline vectorSIMDf ceil(const vectorSIMDf& a)
    {
        vectorSIMDf b = a;
        vector4db_SIMD notTooLarge = b.getAbsoluteValue()<vectorSIMDf(float(0x800000)); //cutoff point for flooring
        __m128i xmm0 = _mm_cvtps_epi32(b.getAsRegister());
        __m128 xmm1 = _mm_cvtepi32_ps(xmm0);

        xmm1 =  _mm_add_ps(xmm1, _mm_and_ps(_mm_cmplt_ps(xmm1, a.getAsRegister()), _mm_set1_ps(1.f)));

        _mm_maskmoveu_si128(_mm_castps_si128(xmm1),notTooLarge.getAsRegister(),(char*)b.pointer);
        return b;
    }
    inline vectorSIMDf fract(const vectorSIMDf& a)
    {
        return a-floor(a);
    }
    inline vectorSIMDf sqrt(const vectorSIMDf& a)
    {
        return _mm_sqrt_ps(a.getAsRegister());
	}
    inline vectorSIMDf inversesqrt(const vectorSIMDf& a)
    {
        return _mm_rsqrt_ps(a.getAsRegister());
	}
    inline vectorSIMDf reciprocal(const vectorSIMDf& a)
    {
        return _mm_rcp_ps(a.getAsRegister());
	}
	inline vectorSIMDf dot(const vectorSIMDf& a, const vectorSIMDf& b)
    {
        __m128 xmm0 = a.getAsRegister();
        __m128 xmm1 = b.getAsRegister();/*
#ifdef __IRR_COMPILE_WITH_SSE4_1
        xmm0 = _mm_dp_ps(xmm0,xmm1,);
#error "Implementation in >=SSE4.1 not ready yet"
#elif __IRR_COMPILE_WITH_SSE3*/
#ifdef __IRR_COMPILE_WITH_SSE3
        xmm0 = _mm_mul_ps(xmm0,xmm1);
        xmm0 = _mm_hadd_ps(xmm0,xmm0);
        return _mm_hadd_ps(xmm0,xmm0);
#elif defined(__IRR_COMPILE_WITH_SSE2)
        xmm0 = _mm_mul_ps(xmm0,xmm1);
        xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(0,1,2,3)));
        return _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(2,3,0,1)));
#endif
    }
	inline vectorSIMDf cross(const vectorSIMDf& a, const vectorSIMDf& b)
    {
        __m128 xmm0 = a.getAsRegister();
        __m128 xmm1 = b.getAsRegister();
#ifdef __IRR_COMPILE_WITH_SSE2 //! SSE2 implementation is faster than previous SSE3 implementation
        __m128 backslash = _mm_mul_ps(FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,0,2,1)),FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,1,0,2)));
        __m128 forwardslash = _mm_mul_ps(FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,1,0,2)),FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,0,2,1)));
        return _mm_sub_ps(backslash,forwardslash); //returns 0 in the last component :D
#endif
    }
    inline vectorSIMDf length(const vectorSIMDf& v)
    {
        __m128 xmm0 = v.getAsRegister();
#ifdef __IRR_COMPILE_WITH_SSE3
        xmm0 = _mm_mul_ps(xmm0,xmm0);
        xmm0 = _mm_hadd_ps(xmm0,xmm0);
        return _mm_sqrt_ps(_mm_hadd_ps(xmm0,xmm0));
#elif defined(__IRR_COMPILE_WITH_SSE2)
        xmm0 = _mm_mul_ps(xmm0,xmm0);
        xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(0,1,2,3)));
        xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(2,3,0,1)));
        return _mm_sqrt_ps(xmm0);
#endif
    }
    inline vectorSIMDf normalize(const vectorSIMDf& v)
    {
        __m128 xmm0 = v.getAsRegister();
        __m128 xmm1 = dot(v,v).getAsRegister();// the uncecessary load/store and variable construction will get optimized out with inline
#ifdef IRRLICHT_FAST_MATH
        return _mm_mul_ps(xmm0,_mm_rsqrt_ps(xmm1));
#else
        return _mm_div_ps(xmm0,_mm_sqrt_ps(xmm1));
#endif
    }

	//! Typedef for a float n-dimensional vector.
	typedef vectorSIMDf vector4df_SIMD;
	typedef vectorSIMDf vector3df_SIMD;
	typedef vectorSIMDf vector2df_SIMD;



    template <class T>
    class vectorSIMD_32 : public SIMD_32bitSwizzleAble<vectorSIMD_32<T>,__m128i>
	{
	public:
		//! Default constructor (null vector).
		inline vectorSIMD_32() {_mm_store_si128((__m128i*)pointer,_mm_setzero_si128());}
		inline vectorSIMD_32(T* const &array) {_mm_store_si128((__m128i*)pointer,_mm_loadu_si128((__m128i*)array));}
		inline vectorSIMD_32(T* const &array, bool ALIGNED) {_mm_store_si128((__m128i*)pointer,_mm_load_si128((__m128i*)array));}
		//! Fastest and most natural constructor
		inline vectorSIMD_32(const __m128i &reg) {_mm_store_si128((__m128i*)pointer,reg);}
		//! Constructor with the same value for all elements
		inline explicit vectorSIMD_32(const T &n) {_mm_store_si128((__m128i*)pointer,_mm_castps_si128(_mm_load_ps1((float*)&n)));}
		//! Copy constructor
		inline vectorSIMD_32(const vectorSIMD_32<T>& other) {_mm_store_si128((__m128i*)pointer,other.getAsRegister());}

/**
        static inline void* operator new(size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void operator delete(void* ptr)
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new[](size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void  operator delete[](void* ptr) throw()
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new(std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete(void* p,void* t) throw() {}
        static inline void* operator new[](std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete[](void* p,void* t) throw() {}
**/

/*
		inline vectorSIMDf(const vectorSIMDu32& other);
		inline vectorSIMDf(const vectorSIMDi32& other);
		inline vectorSIMDf(const vectorSIMDu16& other);
		inline vectorSIMDf(const vectorSIMDi16& other);
**/

		inline vectorSIMD_32<T>& operator=(const vectorSIMD_32<T>& other) { _mm_store_si128((__m128i*)pointer,other.getAsRegister()); return *this; }

        //! bitwise ops
        inline vectorSIMD_32<T> operator&(const vectorSIMD_32<T>& other) {return _mm_and_si128(getAsRegister(),other.getAsRegister());}
        inline vectorSIMD_32<T> operator|(const vectorSIMD_32<T>& other) {return _mm_or_si128(getAsRegister(),other.getAsRegister());}
        inline vectorSIMD_32<T> operator^(const vectorSIMD_32<T>& other) {return _mm_xor_si128(getAsRegister(),other.getAsRegister());}

        //! in case you want to do your own SSE
        inline __m128i getAsRegister() const {return _mm_load_si128((__m128i*)pointer);}

/*
		// operators against vectors
		inline vectorSIMD_32<T> operator-() const { return _mm_xor_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80000000u)),getAsRegister()); }

		inline vectorSIMD_32<T> operator+(const vectorSIMD_32<T>& other) const { return _mm_add_ps(other.getAsRegister(),getAsRegister()); }
		inline vectorSIMD_32<T>& operator+=(const vectorSIMD_32<T>& other) { _mm_store_ps(pointer,_mm_add_ps(other.getAsRegister(),getAsRegister())); return *this; }

		inline vectorSIMD_32<T> operator-(const vectorSIMD_32<T>& other) const { return _mm_sub_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMD_32<T>& operator-=(const vectorSIMD_32<T>& other) { _mm_store_ps(pointer,_mm_sub_ps(getAsRegister(),other.getAsRegister())); return *this; }

		inline vectorSIMDf operator*(const vectorSIMDf& other) const { return _mm_mul_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMD_32<T> operator*(const vectorSIMD_32<T>& other) const { return _mm_mul_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMD_32<T>& operator*=(const vectorSIMD_32<T>& other) { _mm_store_ps(pointer,_mm_mul_ps(getAsRegister(),other.getAsRegister())); return *this; }

		inline vectorSIMDf operator/(const vectorSIMDf& other) const { return preciseDivision(other); }
		inline vectorSIMD_32<T> operator/(const vectorSIMD_32<T>& other) const { return preciseDivision(other); }
		inline vectorSIMD_32<T>& operator/=(const vectorSIMD_32<T>& other) { (*this) = preciseDivision(other); return *this; }

/*
		//operators against scalars
		inline vectorSIMDf  operator+(const float &val) const { return (*this)+vectorSIMDf(val); }
		inline vectorSIMDf& operator+=(const float &val) { return ( (*this) += vectorSIMDf(val) ); }

		inline vectorSIMDf operator-(const float &val) const { return (*this)-vectorSIMDf(val); }
		inline vectorSIMDf& operator-=(const float &val) { return ( (*this) -= vectorSIMDf(val) ); }

		inline vectorSIMDf  operator*(const float &val) const { return (*this)*vectorSIMDf(val); }
		inline vectorSIMDf& operator*=(const float &val) { return ( (*this) *= vectorSIMDf(val) ); }

#ifdef IRRLICHT_FAST_MATH
		inline vectorSIMDf operator/(const float &v) const { return vectorSIMDf(_mm_mul_ps(_mm_rcp_ps(_mm_load_ps1(&v)),getAsRegister())); }
		inline vectorSIMDf& operator/=(const float &v) { _mm_store_ps(pointer,_mm_mul_ps(_mm_rcp_ps(_mm_load_ps1(&v)),getAsRegister())); return *this; }
#else
		inline vectorSIMDf operator/(const float &v) const { return vectorSIMDf(_mm_div_ps(getAsRegister(),_mm_load_ps1(&v))); }
		inline vectorSIMDf& operator/=(const float &v) { _mm_store_ps(pointer,_mm_div_ps(getAsRegister(),_mm_load_ps1(&v))); return *this; }
#endif

		//! I AM BREAKING IRRLICHT'S COMPARISON OPERATORS
		inline vector4db_SIMD operator<=(const vectorSIMDf& other) const
		{
		    return _mm_cmple_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator>=(const vectorSIMDf& other) const
		{
		    return _mm_cmpge_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator<(const vectorSIMDf& other) const
		{
		    return _mm_cmplt_ps(getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator>(const vectorSIMDf& other) const
		{
		    return _mm_cmpgt_ps(getAsRegister(),other.getAsRegister());
		}

		//! only the method that returns bool confirms if two vectors are exactly the same
		inline vectorSIMDf operator==(const vectorSIMDf& other) const
		{
			return _mm_cmpeq_ps(getAsRegister(),other.getAsRegister());
		}
		inline vectorSIMDf operator!=(const vectorSIMDf& other) const
		{
			return _mm_cmpneq_ps(getAsRegister(),other.getAsRegister());
		}



		// functions
		//! zeroes out out of range components (useful before performing a dot product so it doesnt get polluted with random values)
		//! WARNING IT DOES COST CYCLES
		inline void makeSafe2D(void) {_mm_store_ps(pointer,_mm_and_ps(_mm_load_ps(pointer),_mm_castsi128_ps(_mm_set_epi32(0,0,-1,-1))));}
		inline void makeSafe3D(void) {_mm_store_ps(pointer,_mm_and_ps(_mm_load_ps(pointer),_mm_castsi128_ps(_mm_set_epi32(0,-1,-1,-1))));}

		//! slightly faster than memcpy'ing into the pointers
		inline vectorSIMDf& set(float* const &array) {_mm_store_ps(pointer,_mm_loadu_ps(array)); return *this;}
		//! FASTEST WAY TO SET VALUES, Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMDf& set(float* const &array, bool ALIGNED) {_mm_store_ps(pointer,_mm_load_ps(array));}
		//! normal set() like vector3df's, but for different dimensional vectors
		inline vectorSIMDf& set(const float &nx, const float &ny, const float &nz, const float &nw) {_mm_store_ps(pointer,_mm_set_ps(nw,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(const float &nx, const float &ny, const float &nz) {_mm_store_ps(pointer,_mm_set_ps(0.f,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(const float &nx, const float &ny) {_mm_store_ps(pointer,_mm_set_ps(0.f,0.f,ny,nx)); return *this;}
		inline vectorSIMDf& set(const vectorSIMDf& p) {_mm_store_ps(pointer,p.getAsRegister()); return *this;}
		//! convert from vectorNdf types of irrlicht - it will read a few values past the range of the allocated memory but _mm_loadu_ps shouldnt have that kind of protection
		inline vectorSIMDf& set(const vector3df &p) {_mm_store_ps(pointer,_mm_loadu_ps(&p.X)); makeSafe3D(); return *this;}
		inline vectorSIMDf& set(const vector2df &p) {_mm_store_ps(pointer,_mm_loadu_ps(&p.X)); makeSafe2D(); return *this;}

        //! going directly from vectorSIMD to irrlicht types is safe cause vectorSIMDf is wider
		inline vector2df& getAsVector2df(void) const
		{
		    return *((vector2df*)pointer);
		}
		inline vector3df& getAsVector3df(void) const
		{
		    return *((vector3df*)pointer);
		}


		//! Get length of the vector.
		inline float getLengthAsFloat() const
		{
		    __m128 xmm0 = getAsRegister();
		    float result;/*
#ifdef __IRR_COMPILE_WITH_SSE4_1
            xmm0 = _mm_dp_ps(xmm0,xmm0,);
		    xmm0 = _mm_sqrt_ps(xmm0);
#error "Implementation in >=SSE4.1 not ready yet"
#elif __IRR_COMPILE_WITH_SSE3*/ /*
#ifdef __IRR_COMPILE_WITH_SSE3
		    xmm0 = _mm_mul_ps(xmm0,xmm0);
		    xmm0 = _mm_hadd_ps(xmm0,xmm0);
		    xmm0 = _mm_sqrt_ps(_mm_hadd_ps(xmm0,xmm0));
		    _mm_store_ss(&result,xmm0);
		    return result;
#elif defined(__IRR_COMPILE_WITH_SSE2)
		    xmm0 = _mm_mul_ps(xmm0,xmm0);
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(0,1,2,3)));
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(2,3,0,1)));
		    xmm0 = _mm_sqrt_ps(xmm0);
		    _mm_store_ss(&result,xmm0);
		    return result;
#endif
        }
        //! Useful when you have to divide a vector by another vector's length (so you dont convert/store to a scalar)
        //! all components are filled with length
        //! if you need something else, you can get the register and shuffle
		inline vectorSIMDf getLength() const
		{
		    __m128 xmm0 = getAsRegister();
#ifdef __IRR_COMPILE_WITH_SSE3
		    xmm0 = _mm_mul_ps(xmm0,xmm0);
		    xmm0 = _mm_hadd_ps(xmm0,xmm0);
		    return _mm_sqrt_ps(_mm_hadd_ps(xmm0,xmm0));
#elif defined(__IRR_COMPILE_WITH_SSE2)
		    xmm0 = _mm_mul_ps(xmm0,xmm0);
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(0,1,2,3)));
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(2,3,0,1)));
		    return _mm_sqrt_ps(xmm0);
#endif
        }

		//! Get the dot product with another vector.
		inline float dotProductAsInt(const vectorSIMDf& other) const
		{
		    float result;
		    __m128 xmm0 = getAsRegister();
		    __m128 xmm1 = other.getAsRegister();/*
#ifdef __IRR_COMPILE_WITH_SSE4_1
            xmm0 = _mm_dp_ps(xmm0,xmm1,);
#error "Implementation in >=SSE4.1 not ready yet"
#elif __IRR_COMPILE_WITH_SSE3*/ /*
#ifdef __IRR_COMPILE_WITH_SSE3
		    xmm0 = _mm_mul_ps(xmm0,xmm1);
		    xmm0 = _mm_hadd_ps(xmm0,xmm0);
		    xmm0 = _mm_hadd_ps(xmm0,xmm0);
		    _mm_store_ss(&result,xmm0);
		    return result;
#elif defined(__IRR_COMPILE_WITH_SSE2)
		    xmm0 = _mm_mul_ps(xmm0,xmm1);
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(0,1,2,3)));
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(2,3,0,1)));
		    _mm_store_ss(&result,xmm0);
		    return result;
#endif
		}
		inline vectorSIMDf dotProduct(const vectorSIMDf& other) const
		{
		    __m128 xmm0 = getAsRegister();
		    __m128 xmm1 = other.getAsRegister();/*
#ifdef __IRR_COMPILE_WITH_SSE4_1
            xmm0 = _mm_dp_ps(xmm0,xmm1,);
#error "Implementation in >=SSE4.1 not ready yet"
#elif __IRR_COMPILE_WITH_SSE3*/ /*
#ifdef __IRR_COMPILE_WITH_SSE3
		    xmm0 = _mm_mul_ps(xmm0,xmm1);
		    xmm0 = _mm_hadd_ps(xmm0,xmm0);
		    return _mm_hadd_ps(xmm0,xmm0);
#elif defined(__IRR_COMPILE_WITH_SSE2)
		    xmm0 = _mm_mul_ps(xmm0,xmm1);
		    xmm0 = _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(0,1,2,3)));
		    return _mm_add_ps(xmm0,FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(2,3,0,1)));
#endif
		}

		//! Get squared length of the vector.
		/** This is useful because it is much faster than getLength().
		\return Squared length of the vector. *
		inline float getLengthSQAsFloat() const
		{
		    float result;
		    _mm_store_ss(&result,dotProduct(*this).getAsRegister());
		    return result;
		}
        //! Useful when you have to divide a vector by another vector's length (so you dont convert/store to a scalar)
		inline vectorSIMDf getLengthSQ() const
		{
		    return dotProduct(*this);
        }


		//! Get distance from another point.
		/** Here, the vector is interpreted as point in 3 dimensional space. *
		inline float getDistanceFromAsFloat(const vectorSIMDf& other) const
		{
		    float result;
		    _mm_store_ss(&result,((*this)-other).getLength().getAsRegister());
		    return result;
		}
        inline vectorSIMDf getDistanceFrom(const vectorSIMDf& other) const
		{
			return ((*this)-other).getLength();
		}

		//! Returns squared distance from another point.
		/** Here, the vector is interpreted as point in 3 dimensional space. *
		inline uint32_t getDistanceFromSQAsFloat(const vectorSIMDf& other) const
		{
		    float result;
		    _mm_store_ss(&result,((*this)-other).getLengthSQ().getAsRegister());
		    return result;
		}
        inline uint32_t getDistanceFromSQ(const vectorSIMDf& other) const
		{
			return ((*this)-other).getLengthSQ();
		}

		//! Calculates the cross product with another vector.
		/** \param p Vector to multiply with.
		\return Crossproduct of this vector with p. *
		inline vectorSIMDf crossProduct(const vectorSIMDf& p) const
		{
		    __m128 xmm0 = getAsRegister();
		    __m128 xmm1 = p.getAsRegister();
#ifdef __IRR_COMPILE_WITH_SSE2 //! SSE2 implementation is faster than previous SSE3 implementation
		    __m128 backslash = _mm_mul_ps(FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,0,2,1)),FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,1,0,2)));
		    __m128 forwardslash = _mm_mul_ps(FAST_FLOAT_SHUFFLE(xmm0,_MM_SHUFFLE(3,1,0,2)),FAST_FLOAT_SHUFFLE(xmm1,_MM_SHUFFLE(3,0,2,1)));
			return _mm_sub_ps(backslash,forwardslash); //returns 0 in the last component :D
#endif
		}

		//! Inverts the vector.
		inline vectorSIMDf& invert()
		{
			_mm_store_ps(pointer,_mm_xor_ps(_mm_castsi128_ps(_mm_set_epi32(0x80000000u,0x80000000u,0x80000000u,0x80000000u)),getAsRegister()));
			return *this;
		}
		//! Returns component-wise absolute value of a
		inline vectorSIMDf abs(const vectorSIMDf& a) const
		{
			return _mm_and_ps(a.getAsRegister(),_mm_castsi128_ps(_mm_set_epi32(0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF)));
		}
		//! Returns component-wise absolute value of itself
		inline vectorSIMDf getAbsoluteValue() const
		{
			return abs(*this);
		}
*/

#ifdef _IRR_WINDOWS_
        __declspec(align(SIMD_ALIGNMENT)) union
#else
        union
#endif
        {
            struct{
                T X; T Y; T Z; T W;
            };
            struct{
                T x; T y; T z; T w;
            };
            struct{
                T r; T g; T b; T a;
            };
            struct{
                T s; T t; T p; T q;
            };
            T pointer[4];
        };
#ifdef _IRR_WINDOWS_
    };
#else
    } __attribute__ ((__aligned__(SIMD_ALIGNMENT)));
#endif
/*
    class vectorSIMDi32 : public vectorSIMD_32<int32_t>
    {
        //! Constructor with four different values, FASTEST IF the values are constant literals
		//yes this is correct usage with _mm_set_**(), due to little endianness the thing gets set in "reverse" order
		inline explicit vectorSIMDi32(const int32_t &nx, const int32_t &ny, const int32_t &nz, const int32_t &nw) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(nw,nz,ny,nx));}
		//! 3d constructor
		inline explicit vectorSIMDi32(const int32_t &nx, const int32_t &ny, const int32_t &nz) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(0,nz,ny,nx));}
		//! 2d constructor
		inline explicit vectorSIMDi32(const int32_t &nx, const int32_t &ny) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(0,0,ny,nx));}
    };

    class vectorSIMDu32 : public vectorSIMD_32<uint32_t>
    {
        //! Constructor with four different values, FASTEST IF the values are constant literals
		//yes this is correct usage with _mm_set_**(), due to little endianness the thing gets set in "reverse" order
		inline explicit vectorSIMDu32(const uint32_t &nx, const uint32_t &ny, const uint32_t &nz, const uint32_t &nw) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32((const int32_t&)nw,(const int32_t&)nz,(const int32_t&)ny,(const int32_t&)nx));}
		//! 3d constructor
		inline explicit vectorSIMDu32(const uint32_t &nx, const uint32_t &ny, const uint32_t &nz) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(0,(const int32_t&)nz,(const int32_t&)ny,(const int32_t&)nx));}
		//! 2d constructor
		inline explicit vectorSIMDu32(const uint32_t &nx, const uint32_t &ny) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(0,0,(const int32_t&)ny,(const int32_t&)nx));}
    };

/*
    inline vectorSIMDi32 mix(const vectorSIMDi32& a, const vectorSIMDi32& b, const vectorSIMDf& t)
    {
        return a+(b-a)*t;
    }
    inline vectorSIMDi32 lerp(const vectorSIMDi32& a, const vectorSIMDi32& b, const vectorSIMDf& t)
    {
        return mix(a,b,t);
	}
	template<>
	inline vectorSIMDi32 max_(const vectorSIMDi32& a, const vectorSIMDi32& b)
    {
        return _mm_max_ps(a.getAsRegister(),b.getAsRegister());
    }
	template<>
	inline vectorSIMDi32 min_(const vectorSIMDi32& a, const vectorSIMDi32& b)
    {
        return _mm_min_ps(a.getAsRegister(),b.getAsRegister());
    }
    inline vectorSIMDi32 clamp(const vectorSIMDi32& value, const vectorSIMDi32& low, const vectorSIMDi32& high)
    {
        return min_(max_(value,low),high);
    }

	//! Typedef for an integer 3d vector.
	typedef vectorSIMDu32 vector4du32_SIMD;
	typedef vectorSIMDu32 vector3du32_SIMD;
	typedef vectorSIMDu32 vector2du32_SIMD;

	typedef vectorSIMDi32 vector4di32_SIMD;
	typedef vectorSIMDi32 vector3di32_SIMD;
	typedef vectorSIMDi32 vector2di32_SIMD;


	typedef vectorSIMDu16 vector8du16_SIMD;
	typedef vectorSIMDu16 vector7du16_SIMD;
	typedef vectorSIMDu16 vector6du16_SIMD;
	typedef vectorSIMDu16 vector5du16_SIMD;
	typedef vectorSIMDu16 vector4du16_SIMD;
	typedef vectorSIMDu16 vector3du16_SIMD;
	typedef vectorSIMDu16 vector2du16_SIMD;

	typedef vectorSIMDi16 vector8di16_SIMD;
	typedef vectorSIMDi16 vector7di16_SIMD;
	typedef vectorSIMDi16 vector6di16_SIMD;
	typedef vectorSIMDi16 vector5di16_SIMD;
	typedef vectorSIMDi16 vector4di16_SIMD;
	typedef vectorSIMDi16 vector3di16_SIMD;
	typedef vectorSIMDi16 vector2di16_SIMD;*/

} // end namespace core
} // end namespace irr

#endif
#endif


