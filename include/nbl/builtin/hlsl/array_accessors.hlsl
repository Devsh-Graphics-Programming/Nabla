#ifndef _NBL_BUILTIN_HLSL_ARRAY_ACCESSORS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_ARRAY_ACCESSORS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>

namespace nbl
{
namespace hlsl
{
template<typename ArrayType, typename ComponentType, typename I = uint32_t>
struct array_get
{
    ComponentType operator()(NBL_CONST_REF_ARG(ArrayType) arr, const I ix) NBL_CONST_MEMBER_FUNC
    {
        return arr[ix];
    }
};

template<typename ArrayType, typename ComponentType, typename I = uint32_t>
struct array_set
{
    void operator()(NBL_REF_ARG(ArrayType) arr, I index, ComponentType val) NBL_CONST_MEMBER_FUNC
    {
        arr[index] = val;
    }
};
}
}

#endif