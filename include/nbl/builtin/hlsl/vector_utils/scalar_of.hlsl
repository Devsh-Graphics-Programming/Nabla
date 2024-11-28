#ifndef _NBL_BUILTIN_HLSL_VECTOR_UTILS_SCALAR_OF_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_UTILS_SCALAR_OF_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

// default specializatin, works only with HLSL vectors
template<typename T>
struct scalar_of
{
    using type = scalar_type_t<T>;
};

template<typename T>
using scalar_of_t = scalar_of<T>::type;

}
}

#endif