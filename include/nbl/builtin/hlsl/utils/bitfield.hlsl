#ifndef _NBL_BUILTIN_HLSL_UTILS_BITFIELD_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_BITFIELD_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace utils
{

template<typename T, uint32_t Offset, uint32_t Bits>
struct BitField
{
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