#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_

#ifndef NBL_GL_KHR_shader_subgroup_arithmetic
#include <nbl/builtin/hlsl/subgroup/basic_portability.hlsl>
#endif

namespace nbl
{
namespace hlsl
{
namespace subgroup
{

#ifdef NBL_GL_KHR_shader_subgroup_arithmetic
namespace native
{

template<class Binop>
struct reduction;
template<class Binop>
struct exclusive_scan;
template<class Binop>
struct inclusive_scan;

}
#endif

namespace portability
{

// PORTABILITY BINOP DECLARATIONS
template<class Binop, class ScratchAccessor>
struct reduction;
template<class Binop, class ScratchAccessor>
struct inclusive_scan;
template<class Binop, class ScratchAccessor>
struct exclusive_scan;

}

template<class Binop>
struct reduction
{
    template<class ScratchAccessor, typename T>
    T operator()(const T x)
    { // REVIEW: Should these extension headers have the GL name?
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        return native::reduction<Binop>()(x);
    #else
        return portability::reduction<Binop,ScratchAccessor>::create()(x);
    #endif
    }
};

template<class Binop>
struct exclusive_scan
{
    template<class ScratchAccessor, typename T>
    T operator()(const T x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        return native::exclusive_scan<Binop>()(x);
    #else
        portability::exclusive_scan<Binop,ScratchAccessor>::create()(x);
    #endif
    }
};

template<class Binop>
struct inclusive_scan
{
    template<class ScratchAccessor, typename T>
    T operator()(const T x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        return native::inclusive_scan<Binop>()(x);
    #else
        portability::inclusive_scan<Binop,ScratchAccessor>::create()(x);
    #endif
    }
};

}
}
}

#include <nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl>

#endif