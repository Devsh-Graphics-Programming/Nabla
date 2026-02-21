#ifndef _NBL_BUILTIN_HLSL_PORTABLE_INT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_INT64_T_INCLUDED_

#include <nbl/builtin/hlsl/emulated/int64_t.hlsl>
#include <nbl/builtin/hlsl/device_capabilities_traits.hlsl>

// define NBL_FORCE_EMULATED_INT_64 to force using emulated int64 types

namespace nbl
{
namespace hlsl
{
#ifdef __HLSL_VERSION
#ifdef NBL_FORCE_EMULATED_INT_64
template<typename device_caps = void>
using portable_uint64_t = emulated_uint64_t;
template<typename device_caps = void>
using portable_int64_t = emulated_int64_t;
#else
template<typename device_caps = void>
using portable_uint64_t = typename conditional<device_capabilities_traits<device_caps>::shaderInt64, uint64_t, emulated_uint64_t>::type;
template<typename device_caps = void>
using portable_int64_t = typename conditional<device_capabilities_traits<device_caps>::shaderInt64, int64_t, emulated_int64_t>::type;
#endif

#else
template<typename device_caps = void>
using portable_uint64_t = uint64_t;
template<typename device_caps = void>
using portable_int64_t = int64_t;
#endif

}
}

#endif