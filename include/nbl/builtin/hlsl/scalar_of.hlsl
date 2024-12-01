#ifndef _NBL_BUILTIN_HLSL_SCALAR_OF_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCALAR_OF_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO: figure out a better name
template<typename T>
struct scalar_of
{
	using type = scalar_type_t<T>;
};

template<typename T>
using scalar_of_t = typename scalar_of<T>::type;
}
}

#endif