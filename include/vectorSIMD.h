// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_VECTOR_SIMD_H_INCLUDED__
#define __NBL_VECTOR_SIMD_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"

#ifdef __NBL_COMPILE_WITH_X86_SIMD_

#ifdef __NBL_COMPILE_WITH_X86_SIMD_
    #include <nmmintrin.h>
#else
    #error "Check your compiler or project settings for the -m*sse* flag, or upgrade your CPU"
#endif // __NBL_COMPILE_WITH_X86_SIMD_

#include <type_traits>
#include <stdint.h>
#include <math.h>

#include "nbl/core/alloc/AlignedBase.h"
#include "vector2d.h"
#include "vector3d.h"


#define _NBL_VECTOR_ALIGNMENT _NBL_SIMD_ALIGNMENT // if this gets changed to non-16 it can and will break external code


namespace nbl
{

namespace video
{
    class SColor;
}

namespace core
{
    class vectorSIMDIntBase;
	template <class T>
	class vectorSIMD_32;
	template <class T>
	class vectorSIMD_16;
	class vectorSIMDf;

	class NBL_API NBL_FORCE_EBO vectorSIMDIntBase  : public AlignedBase<_NBL_VECTOR_ALIGNMENT>
	{
	    public:
            inline vectorSIMDIntBase() : vectorSIMDIntBase(_mm_setzero_si128()) {}
            //! Copy constructor
            inline vectorSIMDIntBase(const vectorSIMDIntBase& other) {_mm_store_si128((__m128i*)this,other.getAsRegister());}
            //! These constructors will bytewise cast the reg into the value
            inline vectorSIMDIntBase(const __m128i &reg) {_mm_store_si128((__m128i*)this,reg);}

            vectorSIMDIntBase operator&(const vectorSIMDf &other) const;
            vectorSIMDIntBase operator|(const vectorSIMDf &other) const;
            vectorSIMDIntBase operator^(const vectorSIMDf &other) const;

            vectorSIMDIntBase& operator&=(const vectorSIMDf &other);
            vectorSIMDIntBase& operator|=(const vectorSIMDf &other);
            vectorSIMDIntBase& operator^=(const vectorSIMDf &other);

            //! in case you want to do your own SSE
            inline __m128i getAsRegister() const {return _mm_load_si128((__m128i*)this);}
	};

	namespace impl
	{
	    template<class CRTP>
        class NBL_API NBL_FORCE_EBO vectorSIMDIntBase  : public core::vectorSIMDIntBase
        {
                typedef core::vectorSIMDIntBase Base;
            public:
                using Base::Base;

#ifdef _MSC_VER
                // in MSVC default ctor is not inherited?
                vectorSIMDIntBase() : Base() {}
#endif

                //! Copy constructor
                inline vectorSIMDIntBase(const Base& other) : Base(other) {}
                inline vectorSIMDIntBase(const vectorSIMDIntBase<CRTP>& other) : vectorSIMDIntBase(static_cast<const Base&>(other)) {} // delegate

                inline CRTP operator~() const { return _mm_xor_si128(getAsRegister(),_mm_set_epi64x(-0x1ll,-0x1ll)); }

                inline CRTP operator&(const CRTP &other) const { return _mm_and_si128(getAsRegister(),other.getAsRegister()); }
                inline CRTP operator|(const CRTP &other) const { return _mm_or_si128(getAsRegister(),other.getAsRegister()); }
                inline CRTP operator^(const CRTP &other) const { return _mm_xor_si128(getAsRegister(),other.getAsRegister()); }
                inline CRTP operator&(const vectorSIMDf &other) const { return CRTP(Base::operator&(other)); }
                inline CRTP operator|(const vectorSIMDf &other) const { return CRTP(Base::operator|(other)); }
                inline CRTP operator^(const vectorSIMDf &other) const { return CRTP(Base::operator^(other)); }

                inline CRTP& operator&=(const CRTP &other) { _mm_store_si128((__m128i*)this,_mm_and_si128(getAsRegister(),other.getAsRegister())); return *this; }
                inline CRTP& operator|=(const CRTP &other) { _mm_store_si128((__m128i*)this,_mm_or_si128(getAsRegister(),other.getAsRegister())); return *this;}
                inline CRTP& operator^=(const CRTP &other) { _mm_store_si128((__m128i*)this,_mm_xor_si128(getAsRegister(),other.getAsRegister())); return *this;}
                inline CRTP& operator&=(const vectorSIMDf &other) { return static_cast<CRTP&>(Base::operator&=(other));}
                inline CRTP& operator|=(const vectorSIMDf &other) { return static_cast<CRTP&>(Base::operator|=(other));}
                inline CRTP& operator^=(const vectorSIMDf &other) { return static_cast<CRTP&>(Base::operator^=(other));}
        };
	}

#include "SIMDswizzle.h"

	namespace impl
	{
		struct NBL_API NBL_FORCE_EBO empty_base {};
	}

    //a class for bitwise shizz
	template <int components> class NBL_FORCE_EBO vectorSIMDBool : 
		public impl::vectorSIMDIntBase<vectorSIMDBool<components> >, 
		public std::conditional_t<components==4, SIMD_32bitSwizzleAble<vectorSIMDBool<components>, __m128i>, impl::empty_base>
    {
        typedef impl::vectorSIMDIntBase<vectorSIMDBool<components> > Base;
        static_assert(core::isPoT(components)&&components<=16u,"Wrong number of components!\n");
    public:
        using Base::Base;

#ifdef _MSC_VER
        // in MSVC default ctor is not inherited?
        vectorSIMDBool() : Base() {}
#endif

        inline explicit vectorSIMDBool(const __m128 &reg) {_mm_store_ps((float*)this,reg);}
        inline explicit vectorSIMDBool(const __m128d &reg) {_mm_store_pd((double*)this,reg);}

		//! reads 16 bytes from an array of uint8_t
		inline vectorSIMDBool(const uint8_t* const array) : vectorSIMDBool(_mm_loadu_ps((const float*)array)) {}
		//! same as above, BUT WILL CRASH IF ARRAY NOT 16 BYTE ALIGNED
		inline vectorSIMDBool(const uint8_t* const array, bool ALIGNED) : vectorSIMDBool(_mm_load_ps((const float*)array)) {}
		//! Constructor with the same value for all elements
		inline explicit vectorSIMDBool(bool n)  : vectorSIMDBool(n ? _mm_set_epi64x(-0x1ll,-0x1ll):_mm_setzero_si128()) {}

		inline vectorSIMDBool& operator=(const vectorSIMDBool& other) { _mm_store_si128((__m128i*)value,other.getAsRegister()); return *this; }

        /*
        NO BITSHIFTING SUPPORT
        */
		inline vectorSIMDBool<components> operator!() const { assert(false); return vectorSIMDBool<components>(); }
		inline vectorSIMDBool<components> operator&&(const vectorSIMDBool<components> &other) const { assert(false); return vectorSIMDBool<components>(); }
		inline vectorSIMDBool<components> operator||(const vectorSIMDBool<components> &other) const { assert(false); return vectorSIMDBool<components>(); }


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
            assert(false);
            return false;
        }

        uint8_t value[16];
    };

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
    }*/

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
        vectorSIMDBool<4> result(_mm_or_si128(_mm_cmpeq_epi32(other.getAsRegister(),_mm_setzero_si128()),_mm_cmpeq_epi32(getAsRegister(),_mm_setzero_si128())));
        return ~result;
    }
    template <>
	inline vectorSIMDBool<4> vectorSIMDBool<4>::operator||(const vectorSIMDBool<4> &other) const
    {
        __m128i xmm0 = _mm_or_si128(other.getAsRegister(),getAsRegister());
        return ~vectorSIMDBool<4>(_mm_cmpeq_epi32(xmm0,_mm_setzero_si128()));
    }
    template <>
	inline vectorSIMDBool<4> vectorSIMDBool<4>::operator!() const
    {
        return vectorSIMDBool<4>(_mm_cmpeq_epi32(getAsRegister(),_mm_setzero_si128()));
    }
    template <>
	inline vectorSIMDBool<8> vectorSIMDBool<8>::operator&&(const vectorSIMDBool<8> &other) const
    {
        vectorSIMDBool<8> result(_mm_or_si128(_mm_cmpeq_epi16(other.getAsRegister(),_mm_setzero_si128()),_mm_cmpeq_epi16(getAsRegister(),_mm_setzero_si128())));
        return ~result;
    }
    template <>
	inline vectorSIMDBool<8> vectorSIMDBool<8>::operator||(const vectorSIMDBool<8> &other) const
    {
        __m128i xmm0 = _mm_or_si128(other.getAsRegister(),getAsRegister());
        return ~vectorSIMDBool<8>(_mm_cmpeq_epi16(xmm0,_mm_setzero_si128()));
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


#ifdef __GNUC__
// warning: ignoring attributes on template argument __m128i {aka __vector(2) long long int} [-Wignored-attributes] (etc...)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

    template <class T>
    class NBL_API NBL_FORCE_EBO vectorSIMD_32 : public SIMD_32bitSwizzleAble<vectorSIMD_32<T>,__m128i>, public impl::vectorSIMDIntBase<vectorSIMD_32<T> >
	{
        typedef impl::vectorSIMDIntBase<vectorSIMD_32<T> > Base;

		#define COMPARISON_DISPATCH(func_epi32,func_epi16,func_epi8,LEFT,RIGHT) \
			__m128i a = LEFT; \
			__m128i b = RIGHT; \
			if constexpr(!std::is_signed<T>::value) \
			{ \
				__m128i mask; \
				if constexpr(std::is_same<T,uint32_t>::value) \
				{ \
					mask = _mm_set1_epi32(-0x80000000); \
				} \
				else \
				{ \
					if constexpr(std::is_same<T,uint16_t>::value) \
					{ \
						mask = _mm_set1_epi16(-0x8000); \
					} \
					else \
					{ \
						static_assert(!std::is_same<T,uint8_t>::value,"unimplemented for types other than uint{8,16,32}_t"); \
						mask = _mm_set1_epi8(-0x80); \
					} \
				} \
				a = _mm_xor_si128(mask,LEFT); \
				b = _mm_xor_si128(mask,RIGHT); \
			} \
			__m128i result; \
			if constexpr(sizeof(T)==4u) \
			{ \
				result = func_epi32(a,b); \
			} \
			else \
			{ \
				if constexpr(sizeof(T)==2u) \
				{ \
					result = func_epi16(a,b); \
				} \
				else \
				{ \
					static_assert(sizeof(T)!=1u,"unimplemented for types other than {u}int{8,16,32}_t"); \
					result = func_epi8(a,b); \
				} \
			} \
			return result

	public:
	    using Base::Base;
#ifdef _MSC_VER
        // in MSVC default ctor is not inherited?
        vectorSIMD_32() : Base() {}
#endif
		inline vectorSIMD_32(const vectorSIMD_32& other) : Base() {operator=(other);}

        	//! Constructor with four different values, FASTEST IF the values are constant literals
		//yes this is correct usage with _mm_set_**(), due to little endianness the thing gets set in "reverse" order
		inline explicit vectorSIMD_32(T nx, T ny, T nz, T nw) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(nw,nz,ny,nx));}
		//! 3d constructor
		inline explicit vectorSIMD_32(T nx, T ny, T nz) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(0,nz,ny,nx));}
		//! 2d constructor
		inline explicit vectorSIMD_32(T nx, T ny) {_mm_store_si128((__m128i*)pointer,_mm_set_epi32(0,0,ny,nx));}
		//! Fast Constructor from ints, they come in normal order [0]=X,[1]=Y, etc.
		inline vectorSIMD_32(const T* const array) {_mm_store_si128((__m128i*)pointer,_mm_loadu_si128((const __m128i*)array));}
		//! Fastest Constructor from ints, they come in normal order [0]=X,[1]=Y, etc.
		//! Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMD_32(const T* const array, bool ALIGNED) {_mm_store_si128((__m128i*)pointer,_mm_load_si128((const __m128i*)array));}
		//! Fastest and most natural constructor
		inline vectorSIMD_32(const __m128i& reg) { _mm_store_si128((__m128i*)pointer, reg); }
		//! Constructor with the same value for all elements
		inline explicit vectorSIMD_32(T n) {_mm_store_si128((__m128i*)pointer,_mm_castps_si128(_mm_load_ps1((const float*)&n)));}


		inline explicit vectorSIMD_32(const vectorSIMDf& other);

/*
		inline vectorSIMDf(const vectorSIMDu32& other);
		inline vectorSIMDf(const vectorSIMDi32& other);
		inline vectorSIMDf(const vectorSIMDu16& other);
		inline vectorSIMDf(const vectorSIMDi16& other);
**/

		inline T& operator[](size_t ix) { return pointer[ix]; }
		inline const T& operator[](size_t ix) const { return pointer[ix]; }

		inline vectorSIMD_32<T>& operator=(const vectorSIMD_32<T>& other) { _mm_store_si128((__m128i*)pointer,other.getAsRegister()); return *this; }

		// operators against vectors
		inline vectorSIMD_32<T> operator-() const
		{
		    return this->operator~()+vectorSIMD_32<T>(static_cast<T>(1));
        }

		inline vectorSIMD_32<T> operator+(const vectorSIMD_32<T>& other) const { return _mm_add_epi32(other.getAsRegister(),Base::getAsRegister()); }
		inline vectorSIMD_32<T>& operator+=(const vectorSIMD_32<T>& other) { return operator=(operator+(other)); }

		inline vectorSIMD_32<T> operator-(const vectorSIMD_32<T>& other) const { return _mm_sub_epi32(Base::getAsRegister(),other.getAsRegister()); }
		inline vectorSIMD_32<T>& operator-=(const vectorSIMD_32<T>& other) { return operator=(operator-(other)); }

		// TODO: these are messed up (they care about past the vector)
		inline vectorSIMD_32<T> operator*(const vectorSIMD_32<T>& other) const
		{
			// "but since it only stores the lower 32bits, it's really a sign-oblivious instruction that you can use for both"
			return _mm_mullo_epi32(Base::getAsRegister(),other.getAsRegister());
		}
		inline vectorSIMD_32<T>& operator*=(const vectorSIMD_32<T>& other)
		{
			return operator=(operator*(other));
		}

		inline vectorSIMD_32<T> operator/(const vectorSIMD_32<T>& other) const
		{
			// TODO: do something nicer and faster like https://github.com/vectorclass
			return vectorSIMD_32<T>(x / other.x, y / other.y, z / other.z, w / other.w);
		}
		inline vectorSIMD_32<T>& operator/=(const vectorSIMD_32<T>& other)
		{
			return operator=(operator/(other));
		}

		inline vectorSIMD_32<T> operator%(const vectorSIMD_32<T>& other) const
		{
			// TODO: if /= can't be done then %= can't either
			return vectorSIMD_32<T>(x % other.x, y % other.y, z % other.z, w % other.w);
		}
		inline vectorSIMD_32<T>& operator%=(const vectorSIMD_32<T>& other)
		{
			return operator=(operator%(other));
		}

		//operators against scalars
		inline vectorSIMD_32<T>  operator+(T val) const { return operator+(vectorSIMD_32<T>(val)); }
		inline vectorSIMD_32<T>& operator+=(T val) { return operator+=(vectorSIMD_32<T>(val)); }

		inline vectorSIMD_32<T> operator-(T val) const { return operator-(vectorSIMD_32<T>(val)); }
		inline vectorSIMD_32<T>& operator-=(T val) { return operator-=(vectorSIMD_32<T>(val)); }

		// TODO: these are messed up (they care about past the vector)
		inline vectorSIMD_32<T>  operator*(T val) const { return operator*(vectorSIMD_32<T>(val)); }
		friend inline vectorSIMD_32<T> operator*(T val, const vectorSIMD_32<T>& vec) { return vec*val; }
		inline vectorSIMD_32<T>& operator*=(T val) { return operator*=(vectorSIMD_32<T>(val)); }

		inline vectorSIMD_32<T> operator/(T val) const { return operator/(vectorSIMD_32<T>(val)); }
		inline vectorSIMD_32<T>& operator/=(T val) { return operator/=(vectorSIMD_32<T>(val)); }

		inline vectorSIMD_32<T> operator%(T val) const { return operator%(vectorSIMD_32<T>(val)); }
		inline vectorSIMD_32<T>& operator%=(T val) const { return operator%=(vectorSIMD_32<T>(val)); }

		//!
		inline vector4db_SIMD operator<=(const vectorSIMD_32<T>& other) const
		{
			return !operator>(other);
		}
		inline vector4db_SIMD operator>=(const vectorSIMD_32<T>& other) const
		{
			return !operator<(other);
		}
		inline vector4db_SIMD operator<(const vectorSIMD_32<T>& other) const
		{
			COMPARISON_DISPATCH(_mm_cmplt_epi32,_mm_cmplt_epi16,_mm_cmplt_epi8,vectorSIMDIntBase::getAsRegister(),other.getAsRegister());
		}
		inline vector4db_SIMD operator>(const vectorSIMD_32<T>& other) const
		{
			COMPARISON_DISPATCH(_mm_cmpgt_epi32,_mm_cmpgt_epi16,_mm_cmpgt_epi8,vectorSIMDIntBase::getAsRegister(), other.getAsRegister());
		}
	#undef COMPARISON_DISPATCH

		inline vector4db_SIMD operator==(const vectorSIMD_32<T>& other) const
		{
			return vector4db_SIMD(_mm_cmpeq_epi32(vectorSIMDIntBase::getAsRegister(),other.getAsRegister()));
		}
		inline vector4db_SIMD operator!=(const vectorSIMD_32<T>& other) const
		{
			return !operator==(other);
		}


		// functions
		//! zeroes out out of range components (useful before performing a dot product so it doesnt get polluted with random values)
		//! WARNING IT DOES COST CYCLES
		inline void makeSafe2D(void) {_mm_store_si128(pointer,_mm_and_si128(_mm_load_si128(pointer),_mm_set_epi32(0,0,-1,-1)));}
		inline void makeSafe3D(void) {_mm_store_si128(pointer,_mm_and_si128(_mm_load_si128(pointer),_mm_set_epi32(0,-1,-1,-1)));}
/*
		//! slightly faster than memcpy'ing into the pointers
		inline vectorSIMDf& set(float* const &array) {_mm_store_ps(pointer,_mm_loadu_ps(array)); return *this;}
		//! FASTEST WAY TO SET VALUES, Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMDf& set(float* const &array, bool ALIGNED) {_mm_store_ps(pointer,_mm_load_ps(array));}
		//! normal set() like vector3df's, but for different dimensional vectors
		inline vectorSIMDf& set(float nx, float ny, float nz, float nw) {_mm_store_ps(pointer,_mm_set_ps(nw,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(float nx, float ny, float nz) {_mm_store_ps(pointer,_mm_set_ps(0.f,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(float nx, float ny) {_mm_store_ps(pointer,_mm_set_ps(0.f,0.f,ny,nx)); return *this;}
		inline vectorSIMDf& set(const vectorSIMDf& p) {_mm_store_ps(pointer,p.getAsRegister()); return *this;}
		//! convert from vectorNdf types of irrlicht - it will read a few values past the range of the allocated memory but _mm_loadu_ps shouldnt have that kind of protection
		inline vectorSIMDf& set(const vector3df &p) {_mm_store_ps(pointer,_mm_loadu_ps(&p.X)); makeSafe3D(); return *this;}
		inline vectorSIMDf& set(const vector2df &p) {_mm_store_ps(pointer,_mm_loadu_ps(&p.X)); makeSafe2D(); return *this;}
*/
        //! going directly from vectorSIMD to irrlicht types is safe cause vectorSIMDf is wider
		inline vector2di& getAsVector2di(void) const
		{
		    return *((vector2di*)pointer);
		}
		inline vector3di& getAsVector3di(void) const
		{
		    return *((vector3di*)pointer);
		}

        union
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
            //static_assert(std::is_integral<T>::value||sizeof(T)==4u||std::alignment_of<T>::value==4u,"T must be either uint32_t or uint16_t");
        };
    };

    // will have to change these to classes derived from vectorSIMD_32<T>  because of how comparison operators work on signed vs. unsigned
    typedef vectorSIMD_32<uint32_t> vectorSIMDu32;
	typedef vectorSIMD_32<int32_t> vectorSIMDi32;

	//! Typedef for an integer 3d vector.
	typedef vectorSIMDu32 vector4du32_SIMD;
	typedef vectorSIMDu32 vector3du32_SIMD;
	typedef vectorSIMDu32 vector2du32_SIMD;

	static_assert(sizeof(vector4du32_SIMD) == _NBL_VECTOR_ALIGNMENT, "vector4du32_SIMD not same size as uvec4");
	static_assert(alignof(vector4du32_SIMD) == _NBL_VECTOR_ALIGNMENT, "vector4du32_SIMD not same alignment as uvec4");

	typedef vectorSIMDi32 vector4di32_SIMD;
	typedef vectorSIMDi32 vector3di32_SIMD;
	typedef vectorSIMDi32 vector2di32_SIMD;

	static_assert(sizeof(vector4di32_SIMD) == _NBL_VECTOR_ALIGNMENT, "vector4di32_SIMD not same size as ivec4");
	static_assert(alignof(vector4di32_SIMD) == _NBL_VECTOR_ALIGNMENT, "vector4di32_SIMD not same alignment as ivec4");

/*
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
	typedef vectorSIMDi16 vector2di16_SIMD;

	// do we need 8bit vectors?
	*/


    class NBL_API NBL_FORCE_EBO vectorSIMDf : public SIMD_32bitSwizzleAble<vectorSIMDf,__m128>, public AlignedBase<_NBL_VECTOR_ALIGNMENT>
	{
	public:
		//! Default constructor (null vector).
		inline vectorSIMDf() {_mm_store_ps(pointer,_mm_setzero_ps());}
        //! Constructor with four different values, FASTEST IF the values are constant literals
		//yes this is correct usage with _mm_set_**(), due to little endianness the thing gets set in "reverse" order
		inline explicit vectorSIMDf(float nx, float ny, float nz, float nw) {_mm_store_ps(pointer,_mm_set_ps(nw,nz,ny,nx));}
		//! 3d constructor
		inline explicit vectorSIMDf(float nx, float ny, float nz) {_mm_store_ps(pointer,_mm_set_ps(0.f,nz,ny,nx));}
		//! 2d constructor
		inline explicit vectorSIMDf(float nx, float ny) {_mm_store_ps(pointer,_mm_set_ps(0.f,0.f,ny,nx));}
		//! Fast Constructor from floats, they come in normal order [0]=X,[1]=Y, etc.
		inline vectorSIMDf(const float* const &array) {_mm_store_ps(pointer,_mm_loadu_ps(array));}
		//! Fastest Constructor from floats, they come in normal order [0]=X,[1]=Y, etc.
		//! Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMDf(const float* const &array, bool ALIGNED) {_mm_store_ps(pointer,_mm_load_ps(array));}
		//! Fastest and most natural constructor
		inline vectorSIMDf(const __m128 &reg) {_mm_store_ps(pointer,reg);}
		//! Constructor with the same value for all elements
		inline explicit vectorSIMDf(float n) {_mm_store_ps(pointer,_mm_load_ps1(&n));}
		//! Copy constructor
		inline vectorSIMDf(const vectorSIMDf& other) {_mm_store_ps(pointer,other.getAsRegister());}

		//! Conversions
		template<typename T>
		inline explicit vectorSIMDf(const vectorSIMD_32<T>& ivec) : vectorSIMDf(_mm_cvtepi32_ps(ivec.getAsRegister())) {}

/*
		inline vectorSIMDf(const vectorSIMDu32& other);
		inline vectorSIMDf(const vectorSIMDi32& other);
		inline vectorSIMDf(const vectorSIMDu16& other);
		inline vectorSIMDf(const vectorSIMDi16& other);
**/

		inline vectorSIMDf& operator=(const vectorSIMDf& other) { _mm_store_ps(pointer,other.getAsRegister()); return *this; }

        //! bitwise ops
        inline vectorSIMDf operator&(const vectorSIMDIntBase& other) const {return _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(getAsRegister()),other.getAsRegister()));}
        inline vectorSIMDf& operator&=(const vectorSIMDIntBase& other) { return *this = *this & other; };

        inline vectorSIMDf operator|(const vectorSIMDIntBase& other) const {return _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(getAsRegister()),other.getAsRegister()));}
        inline vectorSIMDf& operator|=(const vectorSIMDIntBase& other) { return *this = *this | other; };

        inline vectorSIMDf operator^(const vectorSIMDIntBase& other) const {return _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(getAsRegister()),other.getAsRegister()));}
        inline vectorSIMDf& operator^=(const vectorSIMDIntBase& other) { return *this = *this ^ other; };

		//! getters
		template<typename INT_TYPE>
		inline float& operator[](INT_TYPE index) { return const_cast<float&>(const_cast<const vectorSIMDf*>(this)->pointer[index]); }
		template<typename INT_TYPE>
		inline const float& operator[](INT_TYPE index) const { return pointer[index]; }

        //! in case you want to do your own SSE
        inline __m128 getAsRegister() const {return _mm_load_ps(pointer);}


		// operators against vectors
		inline vectorSIMDf operator-() const { return this->operator^(_mm_set1_epi32(0x80000000u)); }

		inline vectorSIMDf operator+(const vectorSIMDf& other) const { return _mm_add_ps(other.getAsRegister(),getAsRegister()); }
		inline vectorSIMDf& operator+=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_add_ps(other.getAsRegister(),getAsRegister())); return *this; }

		inline vectorSIMDf operator-(const vectorSIMDf& other) const { return _mm_sub_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMDf& operator-=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_sub_ps(getAsRegister(),other.getAsRegister())); return *this; }

		inline vectorSIMDf operator*(const vectorSIMDf& other) const { return _mm_mul_ps(getAsRegister(),other.getAsRegister()); }
		inline vectorSIMDf& operator*=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_mul_ps(getAsRegister(),other.getAsRegister())); return *this; }
#ifdef __NBL_FAST_MATH
		inline vectorSIMDf operator/(const vectorSIMDf& other) const { return _mm_mul_ps(getAsRegister(),_mm_rcp_ps(other.getAsRegister())); }
		inline vectorSIMDf& operator/=(const vectorSIMDf& other) { _mm_store_ps(pointer,_mm_mul_ps(getAsRegister(),_mm_rcp_ps(other.getAsRegister()))); return *this; }
#else
		inline vectorSIMDf operator/(const vectorSIMDf& other) const { return preciseDivision(other); }
		inline vectorSIMDf& operator/=(const vectorSIMDf& other) { (*this) = preciseDivision(other); return *this; }
#endif
		inline vectorSIMDf preciseDivision(const vectorSIMDf& other) const { return _mm_div_ps(getAsRegister(),other.getAsRegister()); }


		//operators against scalars
		inline vectorSIMDf  operator+(float val) const { return (*this)+vectorSIMDf(val); }
		inline vectorSIMDf& operator+=(float val) { return ( (*this) += vectorSIMDf(val) ); }

		inline vectorSIMDf operator-(float val) const { return (*this)-vectorSIMDf(val); }
		inline vectorSIMDf& operator-=(float val) { return ( (*this) -= vectorSIMDf(val) ); }

		inline vectorSIMDf  operator*(float val) const { return (*this)*vectorSIMDf(val); }
		inline vectorSIMDf& operator*=(float val) { return ( (*this) *= vectorSIMDf(val) ); }

#ifdef __NBL_FAST_MATH
		inline vectorSIMDf operator/(float v) const { return vectorSIMDf(_mm_mul_ps(_mm_rcp_ps(_mm_load_ps1(&v)),getAsRegister())); }
		inline vectorSIMDf& operator/=(float v) { _mm_store_ps(pointer,_mm_mul_ps(_mm_rcp_ps(_mm_load_ps1(&v)),getAsRegister())); return *this; }
#else
		inline vectorSIMDf operator/(float v) const { return vectorSIMDf(_mm_div_ps(getAsRegister(),_mm_load_ps1(&v))); }
		inline vectorSIMDf& operator/=(float v) { _mm_store_ps(pointer,_mm_div_ps(getAsRegister(),_mm_load_ps1(&v))); return *this; }
#endif

		//! I AM BREAKING IRRLICHT'S COMPARISON OPERATORS
		inline vector4db_SIMD operator<=(const vectorSIMDf& other) const
		{
		    return vector4db_SIMD(_mm_cmple_ps(getAsRegister(),other.getAsRegister()));
		}
		inline vector4db_SIMD operator>=(const vectorSIMDf& other) const
		{
		    return vector4db_SIMD(_mm_cmpge_ps(getAsRegister(),other.getAsRegister()));
		}
		inline vector4db_SIMD operator<(const vectorSIMDf& other) const
		{
		    return vector4db_SIMD(_mm_cmplt_ps(getAsRegister(),other.getAsRegister()));
		}
		inline vector4db_SIMD operator>(const vectorSIMDf& other) const
		{
		    return vector4db_SIMD(_mm_cmpgt_ps(getAsRegister(),other.getAsRegister()));
		}

		//! only the method that returns bool confirms if two vectors are exactly the same
		inline vector4db_SIMD operator==(const vectorSIMDf& other) const
		{
			return vector4db_SIMD(_mm_cmpeq_ps(getAsRegister(),other.getAsRegister()));
		}
		inline vector4db_SIMD operator!=(const vectorSIMDf& other) const
		{
			return vector4db_SIMD(_mm_cmpneq_ps(getAsRegister(),other.getAsRegister()));
		}



		// functions
		//! zeroes out out of range components (useful before performing a dot product so it doesnt get polluted with random values)
		//! WARNING IT DOES COST CYCLES
		inline void makeSafe2D(void) {this->operator&=(_mm_set_epi32(0,0,-1,-1));}
		inline void makeSafe3D(void) {this->operator&=(_mm_set_epi32(0,-1,-1,-1));}

		//! slightly faster than memcpy'ing into the pointers
		inline vectorSIMDf& set(float* const &array) {_mm_store_ps(pointer,_mm_loadu_ps(array)); return *this;}
		//! FASTEST WAY TO SET VALUES, Address has to be aligned to 16bytes OR WILL CRASH
		inline vectorSIMDf& set(float* const &array, bool ALIGNED) {_mm_store_ps(pointer,_mm_load_ps(array)); return *this;}
		//! normal set() like vector3df's, but for different dimensional vectors
		inline vectorSIMDf& set(float nx, float ny, float nz, float nw) {_mm_store_ps(pointer,_mm_set_ps(nw,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(float nx, float ny, float nz) {_mm_store_ps(pointer,_mm_set_ps(0.f,nz,ny,nx)); return *this;}
		inline vectorSIMDf& set(float nx, float ny) {_mm_store_ps(pointer,_mm_set_ps(0.f,0.f,ny,nx)); return *this;}
		inline vectorSIMDf& set(const vectorSIMDf& p) {_mm_store_ps(pointer,p.getAsRegister()); return *this;}
		//! convert from vectorNdf types of irrlicht
		inline vectorSIMDf& set(const vector3df &p) {_mm_store_ps(pointer,_mm_setr_ps(p.X,p.Y,p.Z,0.f)); return *this;}
		inline vectorSIMDf& set(const vector2df &p) {_mm_store_ps(pointer,_mm_setr_ps(p.X,p.Y,0.f,0.f)); return *this;}

        //! going directly from vectorSIMD to irrlicht types is safe cause vectorSIMDf is wider
		inline vector2df& getAsVector2df(void)
		{
		    return *((vector2df*)pointer);
		}
		inline const vector2df& getAsVector2df(void) const
		{
		    return *((vector2df*)pointer);
		}
		inline vector3df& getAsVector3df(void)
		{
		    return *((vector3df*)pointer);
		}
		inline const vector3df& getAsVector3df(void) const
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

		//! Rotates the vector by a specified number of RADIANS around the Y axis and the specified center.
		/** \param radians Number of RADIANS to rotate around the Y axis.
		\param center The center of the rotation. */
		inline void rotateXZByRAD(float radians, const vectorSIMDf& center=vectorSIMDf())
		{
			vectorSIMDf xmm0 = *this - center;

            float cs = cosf(radians);
            float sn = sinf(radians);
			vectorSIMDf xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0.getAsRegister()); // now contains (X*cos,radom_crap,Z*cos,random_crap)
			vectorSIMDf xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0.getAsRegister(),_MM_SHUFFLE(3,0,1,2))); // now contains (Z*sin,radom_crap,X*cos,random_crap)
			xmm3 ^= _mm_set_epi32(0,0,0,0x80000000u); // invert the Z*sin
			xmm0 = xmm2+xmm3+center; // gives us ((X*cs - Z*sn), (X*cs - Z*sn), (X*sn + Z*cs), (X*sn + Z*cs))

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0.getAsRegister()),_mm_set_epi32(0,-1,0,-1),(char*)pointer);// only overwrites the X,Z elements of our vector
		}

		//! Rotates the vector by a specified number of RADIANS around the Z axis and the specified center.
		/** \param RADIANS: Number of RADIANS to rotate around the Z axis.
		\param center: The center of the rotation. */
		inline void rotateXYByRAD(float radians, const vectorSIMDf& center=vectorSIMDf())
		{
			vectorSIMDf xmm0 = *this - center;

            float cs = cosf(radians);
            float sn = sinf(radians);
			vectorSIMDf xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0.getAsRegister()); // now contains (X*cos,Y*cos,...,...)
			vectorSIMDf xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0.getAsRegister(),_MM_SHUFFLE(3,2,0,1))); // now contains (Y*sin,X*cos,...)
			xmm3 ^= _mm_set_epi32(0,0,0,0x80000000u); // invert the Y*sin
			xmm0 = xmm2 + xmm3 + center; // gives us ((X*cs - Y*sn), (Y*cs + X*sn),...,...)

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0.getAsRegister()),_mm_set_epi32(0,0,-1,-1),(char*)pointer);// only overwrites the X,Y elements of our vector
		}

		//! Rotates the vector by a specified number of degrees around the X axis and the specified center.
		/** \param degrees: Number of degrees to rotate around the X axis.
		\param center: The center of the rotation. */
		inline void rotateYZByRAD(float radians, const vectorSIMDf& center=vectorSIMDf())
		{
			vectorSIMDf xmm0 = *this - center;

            float cs = cosf(radians);
            float sn = sinf(radians);
			vectorSIMDf xmm2 = _mm_mul_ps(_mm_load_ps1(&cs),xmm0.getAsRegister()); // now contains (X*cos,Y*cos,...,...)
			vectorSIMDf xmm3 = _mm_mul_ps(_mm_load_ps1(&sn),FAST_FLOAT_SHUFFLE(xmm0.getAsRegister(),_MM_SHUFFLE(3,1,2,0))); // now contains (...,Z*sin,Y*sin,...)
			xmm3 ^= _mm_set_epi32(0,0,0x80000000u,0); // invert the Z*sin
			xmm0 = xmm2 + xmm3 + center; // gives us ((X*cs - Y*sn), (Y*cs + X*sn),...,...)

            _mm_maskmoveu_si128(_mm_castps_si128(xmm0.getAsRegister()),_mm_set_epi32(0,-1,-1,0),(char*)pointer);// only overwrites the X,Y elements of our vector
		}

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
	};

#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif

	//!
	template<class T>
	vectorSIMD_32<T>::vectorSIMD_32(const vectorSIMDf& other)
	{
		_mm_store_si128(reinterpret_cast<__m128i*>(pointer), _mm_cvtps_epi32(other.getAsRegister()));
	}

	//! Typedef for a float n-dimensional vector.
	typedef vectorSIMDf vector4df_SIMD;
	typedef vectorSIMDf vector3df_SIMD;
	typedef vectorSIMDf vector2df_SIMD;

	static_assert(sizeof(vector4df_SIMD) == _NBL_VECTOR_ALIGNMENT, "vector4df_SIMD not same size as vec4");
	static_assert(alignof(vector4df_SIMD) == _NBL_VECTOR_ALIGNMENT, "vector4df_SIMD not same alignment as vec4");


    inline vectorSIMDIntBase vectorSIMDIntBase::operator&(const vectorSIMDf &other) const { return _mm_and_si128(getAsRegister(),_mm_castps_si128(other.getAsRegister())); }
    inline vectorSIMDIntBase vectorSIMDIntBase::operator|(const vectorSIMDf &other) const { return _mm_or_si128(getAsRegister(),_mm_castps_si128(other.getAsRegister())); }
    inline vectorSIMDIntBase vectorSIMDIntBase::operator^(const vectorSIMDf &other) const { return _mm_xor_si128(getAsRegister(),_mm_castps_si128(other.getAsRegister())); }

    inline vectorSIMDIntBase& vectorSIMDIntBase::operator&=(const vectorSIMDf &other) { _mm_store_si128((__m128i*)this,_mm_and_si128(getAsRegister(),_mm_castps_si128(other.getAsRegister()))); return *this; }
    inline vectorSIMDIntBase& vectorSIMDIntBase::operator|=(const vectorSIMDf &other) { _mm_store_si128((__m128i*)this,_mm_or_si128(getAsRegister(),_mm_castps_si128(other.getAsRegister()))); return *this;}
    inline vectorSIMDIntBase& vectorSIMDIntBase::operator^=(const vectorSIMDf &other) { _mm_store_si128((__m128i*)this,_mm_xor_si128(getAsRegister(),_mm_castps_si128(other.getAsRegister()))); return *this;}




#ifdef __GNUC__
	// warning: ignoring attributes on template argument __m128i {aka __vector(2) long long int} [-Wignored-attributes] (etc...)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

	template <>
	template <int mask>
	inline __m128 SIMD_32bitSwizzleAble<vectorSIMDf, __m128>::shuffleFunc(__m128 reg) const
	{
		return FAST_FLOAT_SHUFFLE(reg, mask);
	}

	template <>
	template <int mask>
	inline __m128i SIMD_32bitSwizzleAble<vectorSIMD_32<int32_t>, __m128i>::shuffleFunc(__m128i reg) const
	{
		return _mm_shuffle_epi32(reg, mask);
	}

	template <>
	template <int mask>
	inline __m128i SIMD_32bitSwizzleAble<vectorSIMD_32<uint32_t>, __m128i>::shuffleFunc(__m128i reg) const
	{
		return _mm_shuffle_epi32(reg, mask);
	}

	template <>
	template <int mask>
	inline __m128i SIMD_32bitSwizzleAble<vectorSIMDBool<4>, __m128i>::shuffleFunc(__m128i reg) const
	{
		return _mm_shuffle_epi32(reg, mask);
	}

#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif


	template <class T, class X>
	class NBL_API NBL_FORCE_EBO SIMD_8bitSwizzleAble
	{
		template<size_t A, size_t B, size_t C, size_t D, size_t E, size_t F, size_t G, size_t H, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N, size_t O, size_t P>
		inline T swizzle() const
		{
			__m128i mask = _mm_set_epi8(P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
			return T(_mm_shuffle_epi8(((const T*)this)->getAsRegister(), mask));
		}
	};

	template <class T, class X>
	class NBL_API NBL_FORCE_EBO SIMD_16bitSwizzleAble
	{
		template<size_t A, size_t B, size_t C, size_t D, size_t E, size_t F, size_t G, size_t H>
		inline T swizzle() const
		{
			__m128i mask = _mm_setr_epi8(2 * A, 2 * A + 1, 2 * B, 2 * B + 1, 2 * C, 2 * C + 1, 2 * D, 2 * D + 1, 2 * E, 2 * E + 1, 2 * F, 2 * F + 1, 2 * G, 2 * G + 1, 2 * H, 2 * H + 1);
			return T(_mm_shuffle_epi8(((const T*)this)->getAsRegister(), mask));
		}
	};

} // end namespace core
} // end namespace nbl

#endif
#endif


