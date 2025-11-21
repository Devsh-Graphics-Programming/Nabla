#ifndef _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

// TODO: why cant I NBL_CONST_REF_ARG(vector<T, N>)
template<typename T, uint32_t N>
inline T lengthsquared(vector<T, N> vec)
{
	return dot(vec, vec);
}

}
}

#endif