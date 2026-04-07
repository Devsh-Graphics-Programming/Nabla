#ifndef _NBL_BUILTIN_HLSL_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TRANSFORMATION_MATRIX_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/quaternions.hlsl>
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

namespace nbl
{
namespace hlsl
{

//! Return true when the three basis vectors form an orthonormal basis within `epsilon`.
template<typename T, typename E = double>
bool isOrthoBase(const T& x, const T& y, const T& z, const E epsilon = 1e-6)
{
	auto isNormalized = [](const auto& v, const auto& epsilon) -> bool
	{
		return hlsl::abs(hlsl::length(v) - static_cast<E>(1.0)) <= epsilon;
	};

	auto isOrthogonal = [](const auto& a, const auto& b, const auto& epsilon) -> bool
	{
		return hlsl::abs(hlsl::dot(a, b)) <= epsilon;
	};

	return isNormalized(x, epsilon) && isNormalized(y, epsilon) && isNormalized(z, epsilon) &&
		isOrthogonal(x, y, epsilon) && isOrthogonal(x, z, epsilon) && isOrthogonal(y, z, epsilon);
}

template<typename T>
matrix<T, 4, 4> getMatrix3x4As4x4(const matrix<T, 3, 4>& mat)
{
	matrix<T, 4, 4> output;
	for (int i = 0; i < 3; ++i)
		output[i] = mat[i];
	output[3] = vector<T, 4>(T(0), T(0), T(0), T(1));

	return output;
}

template<typename T>
matrix<T, 4, 4> getMatrix3x3As4x4(const matrix<T, 3, 3>& mat)
{
	matrix<T, 4, 4> output;
	for (int i = 0; i < 3; ++i)
		output[i] = vector<T, 4>(mat[i], T(1));
	output[3] = vector<T, 4>(T(0), T(0), T(0), T(1));

	return output;
}

template<typename Tout, typename Tin, uint32_t N>
inline vector<Tout, N> getCastedVector(const vector<Tin, N>& in)
{
	vector<Tout, N> out;

	for (int i = 0; i < N; ++i)
		out[i] = (Tout)(in[i]);

	return out;
}

template<typename Tout, typename Tin, uint32_t N, uint32_t M>
inline matrix<Tout, N, M> getCastedMatrix(const matrix<Tin, N, M>& in)
{
	matrix<Tout, N, M> out;

	for (int i = 0; i < N; ++i)
		out[i] = getCastedVector<Tout>(in[i]);

	return out;
}

//! multiplies matrices a and b, 3x4 matrices are treated as 4x4 matrices with 4th row set to (0, 0, 0 ,1)
template<typename T>
inline matrix<T, 3, 4> concatenateBFollowedByA(const matrix<T, 3, 4>& a, const matrix<T, 3, 4>& b)
{
	const auto a4x4 = getMatrix3x4As4x4(a);
	const auto b4x4 = getMatrix3x4As4x4(b);
	return matrix<T, 3, 4>(mul(a4x4, b4x4));
}

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

//! Replace the current rotation and scale by `quat`, leaving translation unchanged.
template<typename T, uint32_t N>
inline void setRotation(matrix<T, N, 4>& outMat, NBL_CONST_REF_ARG(math::quaternion<T>) quat)
{
	static_assert(N == 3 || N == 4);
	matrix<T, 3, 3> mat = _static_cast<matrix<T, 3, 3>>(quat);

	outMat[0] = mat[0];

	outMat[1] = mat[1];

	outMat[2] = mat[2];
}

//! Replace the current translation, leaving the linear part unchanged.
template<typename T, uint32_t N>
inline void setTranslation(matrix<T, N, 4>& outMat, NBL_CONST_REF_ARG(vector<T, 3>) translation)
{
	static_assert(N == 3 || N == 4);

	outMat[0].w = translation.x;
	outMat[1].w = translation.y;
	outMat[2].w = translation.z;
}


}
}

#endif
