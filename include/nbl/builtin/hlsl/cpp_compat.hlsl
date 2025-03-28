#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
// it includes vector and matrix
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>

// Had to push some stuff here to avoid circular dependencies
#include <nbl/builtin/hlsl/cpp_compat/impl/vector_impl.hlsl>

#endif