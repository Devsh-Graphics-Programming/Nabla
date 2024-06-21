#ifndef _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

// TODO: delete
#ifdef __HLSL_VERSION
#define staticAssertTmp(...) ;
#else
void dbgBreakIf(bool condition)
{
	if (!condition)
		__debugbreak();
}
#define staticAssertTmp(x, ...) dbgBreakIf(x);
#endif

namespace nbl
{
namespace hlsl
{
namespace ieee754
{
namespace impl
{
	template <typename T>
	NBL_CONSTEXPR_STATIC_INLINE bool isTypeAllowed()
	{
		return is_same<T, uint16_t>::value ||
			is_same<T, uint32_t>::value ||
			is_same<T, uint64_t>::value ||
			is_same<T, float16_t>::value ||
			is_same<T, float32_t>::value ||
			is_same<T, float64_t>::value;
	}

	template <typename T>
	typename unsigned_integer_of_size<sizeof(T)>::type castToUintType(T x)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return bit_cast<AsUint, T>(x);
	}
	// to avoid bit cast from uintN_t to uintN_t
	template <> unsigned_integer_of_size<2>::type castToUintType(uint16_t x) { return x; }
	template <> unsigned_integer_of_size<4>::type castToUintType(uint32_t x) { return x; }
	template <> unsigned_integer_of_size<8>::type castToUintType(uint64_t x) { return x; }

	template <typename T>
	T castBackToFloatType(T x)
	{
		using AsFloat = typename float_of_size<sizeof(T)>::type;
		return bit_cast<AsFloat, T>(x);
	}
	template<> uint16_t castBackToFloatType(uint16_t x) { return x; }
	template<> uint32_t castBackToFloatType(uint32_t x) { return x; }
	template<> uint64_t castBackToFloatType(uint64_t x) { return x; }
}

	template <typename T>
	int getExponentBitCnt() { staticAssertTmp(false); return 0xdeadbeefu; }
	template<> int getExponentBitCnt<float16_t>() { return 5; }
	template<> int getExponentBitCnt<float32_t>() { return 8; }
	template<> int getExponentBitCnt<float64_t>() { return 11; }

	template <typename T>
	int getMantissaBitCnt() { staticAssertTmp(false); return 0xdeadbeefu; /*TODO: add static assert fail to every 0xdeadbeef, fix all static asserts*/ }
	template<> int getMantissaBitCnt<float16_t>() { return 10; }
	template<> int getMantissaBitCnt<float32_t>() { return 23; }
	template<> int getMantissaBitCnt<float64_t>() { return 52; }

	template <typename T>
	int getExponentBias()
	{
		staticAssertTmp(impl::isTypeAllowed<T>(), "Invalid type! Only floating point or unsigned integer types are allowed.");
		return (0x1 << (getExponentBitCnt<typename float_of_size<sizeof(T)>::type>() - 1)) - 1;
	}

	template<typename T>
	typename unsigned_integer_of_size<sizeof(T)>::type getSignMask()
	{
		staticAssertTmp(impl::isTypeAllowed<T>(), "Invalid type! Only floating point or unsigned integer types are allowed.");
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return AsUint(0x1) << (sizeof(T) * 4 - 1);
	}

	template <typename T>
	typename unsigned_integer_of_size<sizeof(T)>::type getExponentMask() { staticAssertTmp(false); return 0xdeadbeefu; }
	template<> unsigned_integer_of_size<2>::type getExponentMask<float16_t>() { return 0x7C00; }
	template<> unsigned_integer_of_size<4>::type getExponentMask<float32_t>() { return 0x7F800000u; }
	template<> unsigned_integer_of_size<8>::type getExponentMask<float64_t>() { return 0x7FF0000000000000ull; }

	template <typename T>
	typename  unsigned_integer_of_size<sizeof(T)>::type getMantissaMask() { staticAssertTmp(false); return 0xdeadbeefu; }
	template<> unsigned_integer_of_size<2>::type getMantissaMask<float16_t>() { return 0x03FF; }
	template<> unsigned_integer_of_size<4>::type getMantissaMask<float32_t>() { return 0x007FFFFFu; }
	template<> unsigned_integer_of_size<8>::type getMantissaMask<float64_t>() { return 0x000FFFFFFFFFFFFFull; }

	template <typename T>
	uint32_t extractBiasedExponent(T x)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return glsl::bitfieldExtract<AsUint>(impl::castToUintType(x), getMantissaBitCnt<typename float_of_size<sizeof(T)>::type>(), getExponentBitCnt<typename float_of_size<sizeof(T)>::type>());
	}

	template <typename T>
	int extractExponent(T x)
	{
		return int(extractBiasedExponent(x)) - int(getExponentBias<T>());
	}

	template <typename T>
	T replaceBiasedExponent(T x, typename unsigned_integer_of_size<sizeof(T)>::type biasedExp)
	{
		staticAssertTmp(impl::isTypeAllowed<T>(), "Invalid type! Only floating point or unsigned integer types are allowed.");
		return impl::castBackToFloatType<T>(glsl::bitfieldInsert(impl::castToUintType(x), biasedExp, getMantissaBitCnt<T>(), getExponentBitCnt<T>()));
	}

	// performs no overflow tests, returns x*exp2(n)
	template <typename T>
	T fastMulExp2(T x, int n)
	{
		return replaceBiasedExponent(x, extractBiasedExponent(x) + uint32_t(n));
	}

	template <typename T>
	typename unsigned_integer_of_size<sizeof(T)>::type extractMantissa(T x)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return impl::castToUintType(x) & getMantissaMask<typename float_of_size<sizeof(T)>::type>();
	}
}
}
}

#endif