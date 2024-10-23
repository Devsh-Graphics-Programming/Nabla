// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_CORE_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

#ifndef __HLSL_VERSION
#include <vector3d.h>
#endif

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

template<typename T>
inline T radians(NBL_CONST_REF_ARG(T) degrees)
{
	static_assert(
		is_floating_point<T>::value,
		"This code expects the type to be either a double or a float."
	);

	return degrees * PI<T>() / T(180);
}

template<typename T>
inline T degrees(NBL_CONST_REF_ARG(T) radians)
{
	static_assert(
		is_floating_point<T>::value,
		"This code expects the type to be either a double or a float."
	);

	return radians * T(180) / PI<T>();
}

template<typename T>
inline bool equals(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b, NBL_CONST_REF_ARG(T) tolerance)
{
	return (a + tolerance >= b) && (a - tolerance <= b);
}

#ifndef __HLSL_VERSION

NBL_FORCE_INLINE bool equals(const core::vector3df& a, const core::vector3df& b, const core::vector3df& tolerance)
{
	auto ha = a + tolerance;
	auto la = a - tolerance;
	return ha.X >= b.X && ha.Y >= b.Y && ha.Z >= b.Z && la.X <= b.X && la.Y <= b.Y && la.Z <= b.Z;
}

#endif

}
}
}

#endif
