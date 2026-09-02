#ifndef _NBL_BUILTIN_HLSL_UTILS_BITFIELD_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_BITFIELD_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace utils
{

template<typename T, uint32_t Offset, uint32_t Bits>
struct BitField
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t BitCount = Bits;

    static T get(T rawData)
	{
		return glsl::bitfieldExtract(rawData, Offset, Bits);
    }

    static T set(T rawData, T value)
	{
		return glsl::bitfieldInsert(rawData, value, Offset, Bits);
    }
};

}
}
}

#endif