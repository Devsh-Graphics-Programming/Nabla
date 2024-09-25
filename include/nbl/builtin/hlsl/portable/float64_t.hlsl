#ifndef _NBL_BUILTIN_HLSL_PORTABLE_FLOAT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_FLOAT64_T_INCLUDED_

#include <nbl/builtin/hlsl/emulated/float64_t.hlsl>
#include <nbl/builtin/hlsl/device_capabilities_traits.hlsl>
namespace nbl
{
namespace hlsl
{
template<typename device_caps = void>
#ifdef __HLSL_VERSION
using portable_float64_t = typename conditional<device_capabilities_traits<device_caps>::shaderFloat64, float64_t, emulated_float64_t<true, true> >::type;
//using portable_float64_t = typename conditional<device_capabilities_traits<device_caps>::shaderFloat64, emulated_float64_t<true, true>, emulated_float64_t<true, true> >::type;
#else
using portable_float64_t = float64_t;
#endif

//static_assert(sizeof(portable_float64_t) == sizeof(float64_t));

}
}

#endif