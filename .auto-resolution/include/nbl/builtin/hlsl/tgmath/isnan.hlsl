#ifndef _NBL_BUILTIN_HLSL_TGMATH_ISNAN_ISINF_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_ISNAN_ISINF_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

namespace nbl
{
namespace hlsl
{
namespace tgmath_impl
{

template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger> && hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isnan_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	UnsignedInteger absVal = val & (hlsl::numeric_limits<UnsignedInteger>::max >> 1);
	return absVal > (ieee754::traits<AsFloat>::specialValueExp << ieee754::traits<AsFloat>::mantissaBitCnt);
}
template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger>&& hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isinf_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	return (val & (~ieee754::traits<AsFloat>::signMask)) == ieee754::traits<AsFloat>::inf;
}

}
}
}

#endif