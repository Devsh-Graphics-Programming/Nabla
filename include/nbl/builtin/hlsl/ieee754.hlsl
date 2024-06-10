#ifndef _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl::hlsl::ieee754
{
	template <typename T>
	int getExponentBitCnt() { return 0xdeadbeefu; }
	template<> int getExponentBitCnt<float16_t>() { return 5; }
	template<> int getExponentBitCnt<uint16_t>() { return 5; }
	template<> int getExponentBitCnt<float>() { return 8; }
	template<> int getExponentBitCnt<uint32_t>() { return 8; }
	template<> int getExponentBitCnt<double>() { return 11; }
	template<> int getExponentBitCnt<uint64_t>() { return 11; }

	template <typename T>
	int getMantissaBitCnt() { return 0xdeadbeefu; }
	template<> int getMantissaBitCnt<float16_t>() { return 10; }
	template<> int getMantissaBitCnt<uint16_t>() { return 10; }
	template<> int getMantissaBitCnt<float>() { return 23; }
	template<> int getMantissaBitCnt<uint32_t>() { return 23; }
	template<> int getMantissaBitCnt<double>() { return 52; }
	template<> int getMantissaBitCnt<uint64_t>() { return 52; }

	template <typename T>
	int getExponentBias() { return 0xdeadbeefu; }
	template<> int getExponentBias<float16_t>() { return 15; }
	template<> int getExponentBias<uint16_t>() { return 15; }
	template<> int getExponentBias<float>() { return 127; }
	template<> int getExponentBias<uint32_t>() { return 127; }
	template<> int getExponentBias<double>() { return 1023; }
	template<> int getExponentBias<uint64_t>() { return 1023; }

	template <typename T>
	unsigned_integer_of_size<sizeof(T)>::type getExponentMask() { return 0xdeadbeefu; }
	template<> unsigned_integer_of_size<2>::type getExponentMask<float16_t>() { return 0x7C00; }
	template<> unsigned_integer_of_size<2>::type getExponentMask<uint16_t>() { return 0x7C00; }
	template<> unsigned_integer_of_size<4>::type getExponentMask<float>() { return 0x7F800000u; }
	template<> unsigned_integer_of_size<4>::type getExponentMask<uint32_t>() { return 0x7F800000u; }
	template<> unsigned_integer_of_size<8>::type getExponentMask<double>() { return 0x7FF0000000000000ull; }
	template<> unsigned_integer_of_size<8>::type getExponentMask<uint64_t>() { return 0x7FF0000000000000ull; }

	template <typename T>
	unsigned_integer_of_size<sizeof(T)>::type getMantissaMask() { return 0xdeadbeefu; }
	template<> unsigned_integer_of_size<2>::type getMantissaMask<float16_t>() { return 0x03FF; }
	template<> unsigned_integer_of_size<2>::type getMantissaMask<uint16_t>() { return 0x03FF; }
	template<> unsigned_integer_of_size<4>::type getMantissaMask<float>() { return 0x007FFFFFu; }
	template<> unsigned_integer_of_size<4>::type getMantissaMask<uint32_t>() { return 0x007FFFFFu; }
	template<> unsigned_integer_of_size<8>::type getMantissaMask<double>() { return 0x000FFFFFFFFFFFFFull; }
	template<> unsigned_integer_of_size<8>::type getMantissaMask<uint64_t>() { return 0x000FFFFFFFFFFFFFull; }

	template <typename T>
	uint32_t extractBiasedExponent(T x)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return bitfieldExtract<AsUint>(bit_cast<AsUint>(x), getMantissaBitCnt<T>(), getExponentBitCnt<T>());
	}

	template <typename T>
	int extractExponent(T x)
	{
		return int(extractBiasedExponent(x) - getExponentBias<T>());
	}

	template <typename T>
	T replaceBiasedExponent(T x, uint32_t biasedExp)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return bitCast<T, AsUint>(uintBitsToFloat(bitfieldInsert(bit_cast<AsUint>(x), biasedExp, getMantissaBitCnt<T>(), getExponentBitCnt<T>())));
	}

	// performs no overflow tests, returns x*exp2(n)
	template <typename T>
	T fastMulExp2(T x, int n)
	{
		return replaceBiasedExponent(x, extractBiasedExponent(x) + uint32_t(n));
	}

	template <typename T>
	unsigned_integer_of_size<sizeof(T)>::type extractMantissa(T x)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return (bit_cast<AsUint, T>(x) & getMantissaMask<T>());
	}
}

#endif