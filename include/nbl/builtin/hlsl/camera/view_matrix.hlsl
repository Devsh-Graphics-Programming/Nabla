#ifndef _NBL_BUILTIN_HLSL_PROJECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_PROJECTION_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

// /Arek: glm:: for normalize till dot product is fixed (ambiguity with glm namespace + linker issues)
template<typename T>
inline matrix<T, 3, 4> buildCameraLookAtMatrixLH(
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
inline matrix<T, 3, 4> buildCameraLookAtMatrixRH(
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

}
}

#endif