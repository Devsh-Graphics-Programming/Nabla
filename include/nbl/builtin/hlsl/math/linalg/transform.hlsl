// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_TRANSFORM_INCLUDED_

#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/math/linalg/basic.hlsl>
#include <nbl/builtin/hlsl/shapes/aabb.hlsl>

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace linalg
{

// /Arek: glm:: for normalize till dot product is fixed (ambiguity with glm namespace + linker issues)
template<typename T>
inline matrix<T, 3, 4> lhLookAt(
    const vector<T, 3>& position,
    const vector<T, 3>& target,
    const vector<T, 3>& upVector)
{
    const vector<T, 3> zaxis = hlsl::normalize(target - position);
    const vector<T, 3> xaxis = hlsl::normalize(hlsl::cross(upVector, zaxis));
    const vector<T, 3> yaxis = hlsl::cross(zaxis, xaxis);

    matrix<T, 3, 4> r;
    r[0] = vector<T, 4>(xaxis, -hlsl::dot(xaxis, position));
    r[1] = vector<T, 4>(yaxis, -hlsl::dot(yaxis, position));
    r[2] = vector<T, 4>(zaxis, -hlsl::dot(zaxis, position));

    return r;
}

template<typename T>
inline matrix<T, 3, 4> rhLookAt(
    const vector<T, 3>& position,
    const vector<T, 3>& target,
    const vector<T, 3>& upVector)
{
    const vector<T, 3> zaxis = hlsl::normalize(position - target);
    const vector<T, 3> xaxis = hlsl::normalize(hlsl::cross(upVector, zaxis));
    const vector<T, 3> yaxis = hlsl::cross(zaxis, xaxis);

    matrix<T, 3, 4> r;
    r[0] = vector<T, 4>(xaxis, -hlsl::dot(xaxis, position));
    r[1] = vector<T, 4>(yaxis, -hlsl::dot(yaxis, position));
    r[2] = vector<T, 4>(zaxis, -hlsl::dot(zaxis, position));

	return r;
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPoint<T>)
inline shapes::AABB<3, T> pseudo_mul(NBL_CONST_REF_ARG(matrix<T, 3, 4>) lhs, NBL_CONST_REF_ARG(shapes::AABB<3, T>) rhs)
{
	const auto translation = hlsl::transpose(lhs)[3];
	auto transformed = shapes::util::transform(lhs, rhs);
	transformed.minVx += translation;
	transformed.maxVx += translation;
	return transformed;
}

}
}
}
}
#endif
