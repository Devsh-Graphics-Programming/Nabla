#ifndef _NBL_BUILTIN_HLSL_IEE754_IMPL_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_IMPL_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ieee754
{

namespace impl
{
template <typename T>
NBL_CONSTEXPR_INLINE_FUNC bool isTypeAllowed()
{
	return is_same<T, uint16_t>::value ||
		is_same<T, uint32_t>::value ||
		is_same<T, uint64_t>::value ||
		is_same<T, float16_t>::value ||
		is_same<T, float32_t>::value ||
		is_same<T, float64_t>::value;
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type bitCastToUintType(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	return bit_cast<AsUint, T>(x);
}
// to avoid bit cast from uintN_t to uintN_t
template <> NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<2>::type bitCastToUintType(uint16_t x) { return x; }
template <> NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<4>::type bitCastToUintType(uint32_t x) { return x; }
template <> NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<8>::type bitCastToUintType(uint64_t x) { return x; }

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC T castBackToFloatType(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return bit_cast<AsFloat, T>(x);
}
template<> NBL_CONSTEXPR_INLINE_FUNC uint16_t castBackToFloatType(uint16_t x) { return x; }
template<> NBL_CONSTEXPR_INLINE_FUNC uint32_t castBackToFloatType(uint32_t x) { return x; }
template<> NBL_CONSTEXPR_INLINE_FUNC uint64_t castBackToFloatType(uint64_t x) { return x; }
}

}
}
}

#endif