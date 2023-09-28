#ifndef _NBL_BUILTIN_HLSL_IEE754_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ieee754
{

template<uint32_t mantissaBits, uint32_t exponentBits>
struct stats
{
	NBL_CONSTEXPR uint32_t exponentMask = ((1u << exponentBits) - 1) << mantissaBits;
	NBL_CONSTEXPR uint32_t mantissaMask = (1u << mantissaBits) - 1;
	NBL_CONSTEXPR uint32_t exponentBias = (1u << (exponentBits - 1)) - 1;
	NBL_CONSTEXPR uint32_t biasedMaxExp = ((1u << exponentBits) - 1) - 1;
	NBL_CONSTEXPR uint32_t maxValueU = (biasedMaxExp << mantissaBits);
	NBL_CONSTEXPR uint32_t minValueU = 1u;
};

template <uint32_t mantissaBits, uint32_t expBits>
float ufloat_to_float(uint32_t uVal)
{
	const uint32_t exponent = (uVal & stats<mantissaBits, expBits>::exponentMask) << (23 - mantissaBits);
	return asfloat(exponent | (uVal & stats<mantissaBits, expBits>::mantissaMask));
}

template<uint32_t mantissaBits, uint32_t expBits>
uint32_t extract_exponent(uint32_t uVal)
{
	return uVal & stats<mantissaBits, expBits>::exponentMask;
}

template<uint32_t mantissaBits, uint32_t expBits>
uint32_t extract_mantissa(uint32_t uVal)
{
	return uVal & stats<mantissaBits, expBits>::mantissaMask;
}

bool isnan(float _f32)
{
	return asuint(_f32) > stats<23,8>::exponentMask;
}

template <uint32_t mantissaBits, uint32_t expBits>
uint32_t encode_ufloat(float _f32)
{
	uint minSinglePrecisionVal = asuint(ufloat_to_float<mantissaBits, expBits>(stats<expBits, mantissaBits>::maxValueU));
	uint maxSinglePrecisionVal = asuint(ufloat_to_float<mantissaBits, expBits>(stats<expBits, mantissaBits>::minValueU));

	if (_f32 < asfloat(maxSinglePrecisionVal))
	{
		if (_f32 < asfloat(minSinglePrecisionVal))
			return 0;

		const uint32_t _uf32 = asuint(_f32);
		const uint32_t exponent = extract_exponent<23, 8>(_uf32);
		const uint32_t mantissa = extract_mantissa<23, 8>(_uf32);

		return (exponent << expBits) | (mantissa & stats<mantissaBits, expBits>::mantissaMask);
	}

	return stats<mantissaBits, expBits>::exponentMask | (isnan(_f32) ? stats<mantissaBits, expBits>::mantissaMask : 0u);
}

}
}
}
#endif