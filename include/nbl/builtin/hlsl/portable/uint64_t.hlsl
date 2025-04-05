#ifndef _NBL_BUILTIN_HLSL_PORTABLE_UINT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_UINT64_T_INCLUDED_

#include <nbl/builtin/hlsl/emulated/uint64_t.hlsl>
#include <nbl/builtin/hlsl/device_capabilities_traits.hlsl>

// define NBL_FORCE_EMULATED_UINT_64 to force using emulated uint64

namespace nbl
{
namespace hlsl
{
template<typename device_caps = void>
#ifdef __HLSL_VERSION
#ifdef NBL_FORCE_EMULATED_UINT_64
using portable_uint64_t = emulated_uint64_t;
#else
using portable_uint64_t = typename conditional<device_capabilities_traits<device_caps>::shaderInt64, uint64_t, emulated_uint64_t>::type;
#endif

#else
using portable_uint64_t = uint64_t;
#endif

//static_assert(sizeof(portable_uint64_t) == sizeof(uint64_t));

}
}

#endif